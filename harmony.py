"""Harmony energy terms and an independent two-colour harmony score.

Energy terms (used for guidance):
  * hue-template deviation in OKLCH (Matsuda templates as used in colour harmonisation)
  * CIEDE2000 separation between text layers and the layer beneath them
  * a lightness-hierarchy prior: -log p(L_i - L_j | role_i, role_j) fitted on training designs

Independent score (used only for evaluation): the Ou-Luo two-colour harmony model.
"""
import math

import numpy as np
import torch

import colorlib as cl

# Matsuda templates: list of (centre offset in degrees, width in degrees)
TEMPLATES = {
    'i': [(0, 18)], 'V': [(0, 93.6)], 'L': [(0, 18), (90, 79.2)], 'I': [(0, 18), (180, 18)],
    'T': [(0, 180)], 'Y': [(0, 93.6), (180, 18)], 'X': [(0, 93.6), (180, 93.6)],
}
ROT = np.deg2rad(np.arange(0, 360, 5.0))
CHROMA_MIN = 0.03          # OKLCH chroma below which a colour is treated as achromatic


def _arcdist(h, c):
    d = np.abs((h - c + np.pi) % (2 * np.pi) - np.pi)
    return d


def sector_distance(h, tmpl, rot):
    """Angular distance (rad) from hue h to the nearest sector of a rotated template."""
    best = None
    for off, w in TEMPLATES[tmpl]:
        c = rot + np.deg2rad(off)
        d = np.maximum(_arcdist(h, c) - np.deg2rad(w) / 2, 0.0)
        best = d if best is None else np.minimum(best, d)
    return best


def fit_template(hues, weights):
    """Return (template, rotation, normalised deviation) minimising weighted deviation (vectorised over rotations)."""
    if weights.sum() <= 1e-9:
        return 'T', 0.0, 0.0
    best = (None, 0.0, 1e9)
    h = np.asarray(hues)[None, :]
    for t, sectors in TEMPLATES.items():
        d = None
        for off, w in sectors:
            c = ROT[:, None] + np.deg2rad(off)
            ds = np.maximum(_arcdist(h, c) - np.deg2rad(w) / 2, 0.0)
            d = ds if d is None else np.minimum(d, ds)
        cost = (d * weights[None]).sum(1) / weights.sum()
        k = int(cost.argmin())
        if cost[k] < best[2] - 1e-12:
            best = (t, ROT[k], float(cost[k]))
    return best


def hue_weights(lab, area):
    lch = cl.oklab_to_oklch(torch.as_tensor(lab, dtype=torch.float32)).numpy()
    w = np.clip((lch[:, 1] - CHROMA_MIN) / 0.1, 0, 1) * np.sqrt(np.maximum(area, 1e-6))
    return lch[:, 2], w


class LightnessPrior:
    """Histogram prior over OKLab lightness offsets for (upper role, lower role) pairs."""
    BINS = np.linspace(-1.0, 1.0, 41)

    def __init__(self, table=None):
        self.table = table or {}

    @classmethod
    def fit(cls, designs):
        counts = {}
        for d in designs:
            n = d['n']
            for i in range(n):
                for j in range(i):
                    if d['ovl'][i, j] < 0.5:
                        continue
                    li = d['col'][i, 0] if d['rec'][i] else (d['fixed'][i, 0] if d['has_fixed'][i] else None)
                    lj = d['col'][j, 0] if d['rec'][j] else (d['fixed'][j, 0] if d['has_fixed'][j] else None)
                    if li is None or lj is None:
                        continue
                    key = (int(d['type'][i]), int(d['type'][j]))
                    counts.setdefault(key, np.ones(len(cls.BINS) - 1))
                    b = np.clip(np.digitize(li - lj, cls.BINS) - 1, 0, len(cls.BINS) - 2)
                    counts[key][b] += 1
        table = {k: (-np.log(v / v.sum())).tolist() for k, v in counts.items()}
        return cls(table)

    def energy(self, key, dl):
        if key not in self.table:
            return np.zeros_like(dl)
        t = np.asarray(self.table[key])
        b = np.clip(np.digitize(dl, self.BINS) - 1, 0, len(self.BINS) - 2)
        return t[b] - t.min()


def softplus(x):
    return np.logaddexp(0, x)


def separation_energy(de, tau=25.0, s=5.0):
    return softplus((tau - de) / s)


# ---------------------------------------------------------------------------
# Ou & Luo (2006) two-colour harmony model, computed in CIELAB (D65).

def ou_luo_harmony(lab1, lab2):
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    h1, h2 = np.degrees(np.arctan2(b1, a1)) % 360, np.degrees(np.arctan2(b2, a2)) % 360
    dh = np.abs(h1 - h2)
    dh = np.where(dh > 180, 360 - dh, dh)
    dH = 2 * np.sqrt(C1 * C2) * np.sin(np.radians(dh) / 2)
    dC = np.sqrt(dH ** 2 + ((C1 - C2) / 1.46) ** 2)
    HC = 0.04 + 0.53 * np.tanh(0.8 - 0.045 * dC)
    HLsum = 0.28 + 0.54 * np.tanh(-3.88 + 0.029 * (L1 + L2))
    HdL = 0.14 + 0.15 * np.tanh(-2 + 0.2 * np.abs(L1 - L2))
    HL = HLsum + HdL

    def hsy(L, C, h):
        EC = 0.5 + 0.5 * np.tanh(-2 + 0.5 * C)
        HS = -0.08 - 0.14 * np.sin(np.radians(h + 50)) - 0.07 * np.sin(np.radians(2 * h + 90))
        u = (90 - h) / 10
        EY = ((0.22 * L - 12.8) / 10) * np.exp(u - np.exp(u))
        return EC * (HS + EY)

    HH = hsy(L1, C1, h1) + hsy(L2, C2, h2)
    return HC + HL + HH
