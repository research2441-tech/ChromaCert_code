"""Accessibility layer: proxy contrast, hue-preserving OKLCH repair, pixel-level
legibility audit under simulated colour-vision deficiency, and conformal risk
control of the repair margin.
"""
import glob
import pickle

import colour
import numpy as np
import torch

import colorlib as cl

T_TEXT = 1
WCAG_BODY, WCAG_LARGE = 4.5, 3.0
APCA_BODY, APCA_LARGE = 60.0, 45.0
SEVERITIES = (0.2, 0.4, 0.6, 0.8, 1.0)
PIX_Q = 0.10        # a text layer fails if more than 10% of its background pixels fail

_KIND = {'protan': 'Protanomaly', 'deutan': 'Deuteranomaly', 'tritan': 'Tritanomaly'}


def conditions(full=True):
    out = [('normal', 0.0, np.eye(3))]
    for k, name in _KIND.items():
        for s in (SEVERITIES if full else (1.0,)):
            out.append((k, s, np.asarray(colour.blindness.matrix_cvd_Machado2009(name, s))))
    return out


COND_FULL = conditions(True)      # 16 observers used by the pixel-level audit
COND_PROXY = conditions(False)    # 4 observers used by the proxy check during repair


def lin(rgb):
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def delin(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def lum_wcag(lin_rgb):
    return lin_rgb @ np.array([0.2126, 0.7152, 0.0722])


def apca_y(srgb):
    y = np.clip(srgb, 0, 1) ** 2.4 @ np.array([0.2126729, 0.7151522, 0.0721750])
    return np.where(y < 0.022, y + (0.022 - np.minimum(y, 0.022)) ** 1.414, y)


def apca_abs(yt, yb):
    normal = (np.maximum(yb, 1e-9) ** 0.56 - np.maximum(yt, 1e-9) ** 0.57) * 1.14
    rev = (np.maximum(yb, 1e-9) ** 0.65 - np.maximum(yt, 1e-9) ** 0.62) * 1.14
    s = np.where(yb > yt, normal, rev)
    out = np.where(np.abs(s) < 0.1, 0.0, np.where(s > 0, s - 0.027, s + 0.027))
    out = np.where(np.abs(yb - yt) < 0.0005, 0.0, out)
    return np.abs(out) * 100


def view(srgb, M):
    """sRGB as seen by an observer with linear-RGB simulation matrix M."""
    return delin(lin(srgb) @ M.T)


def contrast(fg_srgb, bg_srgb, M, metric):
    f, g = view(fg_srgb, M), view(bg_srgb, M)
    if metric == 'wcag':
        lf, lg = lum_wcag(lin(f)), lum_wcag(lin(g))
        return (np.maximum(lf, lg) + 0.05) / (np.minimum(lf, lg) + 0.05)
    return apca_abs(apca_y(f), apca_y(g))


def threshold(r, i, metric):
    large = r['font_px'][i] >= 24
    if metric == 'wcag':
        return WCAG_LARGE if large else WCAG_BODY
    return APCA_LARGE if large else APCA_BODY


def to_srgb(oklab):
    return cl.oklab_to_srgb(torch.as_tensor(np.asarray(oklab), dtype=torch.float32)).numpy()


# --- proxy check and repair --------------------------------------------------

def text_pairs(r):
    return [(i, int(r['bg'][i])) for i in range(r['n']) if r['type'][i] == T_TEXT and r['bg'][i] >= 0 and r['rec'][i]]


def bg_colour(r, cols, j):
    return cols[j] if r['rec'][j] else r['fixed'][j, :3]


def bg_colours(r, cols, j):
    """Layer-level background evidence: a flat colour, or mean/q10/q90 of a fixed layer."""
    if r['rec'][j]:
        return cols[j:j + 1]
    return r['fixed'][j].reshape(3, 3)


def evidence(r, cols, i):
    """Layer-level background evidence under text layer i: every non-text layer beneath it whose
    box covers more than 10% of the text box, as flat colours or mean/q10/q90 summaries."""
    ev = []
    for j in range(i):
        if r['type'][j] == T_TEXT or r['ovl'][i, j] <= 0.1:
            continue
        if r['rec'][j] or r['has_fixed'][j]:
            ev.append(bg_colours(r, cols, j))
    j0 = int(r['bg'][i])
    if not ev and j0 >= 0:
        ev.append(bg_colours(r, cols, j0))
    return np.concatenate(ev) if ev else np.ones((1, 3), np.float32)


def proxy_min_contrast(r, cols, fg_srgb, j, metric, i=None):
    """Worst proxy contrast of candidate foreground colour(s) over 4 observers and the background evidence."""
    bgs = to_srgb(evidence(r, cols, i) if i is not None else bg_colours(r, cols, j))
    fg = np.atleast_2d(fg_srgb)
    out = np.full(len(fg), np.inf)
    for b in bgs:
        for _, _, M in COND_PROXY:
            out = np.minimum(out, contrast(fg, np.broadcast_to(b, fg.shape), M, metric))
    return out


def heterogeneity(r, cols, i):
    """Largest CIEDE2000 distance between any two evidence colours beneath text layer i."""
    ev = to_srgb(evidence(r, cols, i))
    if len(ev) < 2:
        return 0.0
    labs = cl.srgb_to_cielab(torch.tensor(ev, dtype=torch.float32))
    a = labs[:, None].expand(len(labs), len(labs), 3).reshape(-1, 3)
    b = labs[None].expand(len(labs), len(labs), 3).reshape(-1, 3)
    return float(cl.ciede2000(a, b).max())


_L = np.linspace(0.0, 1.0, 101)


def repair(r, cols, lam, metric='wcag', locked=None):
    """Hue-preserving OKLCH repair of text colours against the proxy background.

    Text layers whose proxy contrast is below (1 + lam) x threshold for any of the
    four proxy observers are moved to the closest OKLCH point (same hue) that
    meets the target; chroma shrinks only as far as needed to stay in gamut.
    Returns repaired OKLab colours and the number of edited layers."""
    cols = cols.copy()
    edits = 0
    for i, j in text_pairs(r):
        if locked is not None and locked[i]:
            continue
        thr = threshold(r, i, metric) * (1 + lam)
        fg = to_srgb(cols[i])
        cmin = proxy_min_contrast(r, cols, fg, j, metric)[0]
        if cmin >= thr:
            continue
        lch = cl.oklab_to_oklch(torch.tensor(cols[i], dtype=torch.float32)).numpy()
        cand = []
        for scale in (1.0, 0.75, 0.5, 0.25, 0.0):
            lab = cl.oklch_to_oklab(torch.tensor(np.stack([_L, np.full_like(_L, lch[1] * scale), np.full_like(_L, lch[2])], 1), dtype=torch.float32))
            ok = cl.oklab_in_gamut(lab).numpy()
            cand.append((lab.numpy()[ok], scale))
        labs = np.concatenate([c for c, _ in cand]) if cand else np.zeros((0, 3))
        if len(labs) == 0:
            continue
        rgbs = to_srgb(labs)
        cmins = proxy_min_contrast(r, cols, rgbs, j, metric)
        feas = cmins >= thr
        o_lab = cl.srgb_to_cielab(torch.tensor(fg[None], dtype=torch.float32))
        de = cl.ciede2000(cl.srgb_to_cielab(torch.tensor(rgbs, dtype=torch.float32)), o_lab.expand(len(rgbs), 3)).numpy()
        k = int(np.argmin(np.where(feas, de, np.inf))) if feas.any() else int(np.argmax(cmins))
        cols[i] = labs[k]
        edits += 1
    return cols, edits


def proxy_fail(r, cols, metric='wcag'):
    f = []
    for i, j in text_pairs(r):
        fg = to_srgb(cols[i])
        f.append(proxy_min_contrast(r, cols, fg, j, metric)[0] < threshold(r, i, metric))
    return np.array(f, bool)


# --- pixel-level audit ---------------------------------------------------------

class Thumbs:
    def __init__(self, prep='prep', splits=('validation', 'test')):
        self.t = {}
        for s in splits:
            for f in sorted(glob.glob(f'{prep}/{s}-*_thumbs.pkl')):
                self.t.update(pickle.load(open(f, 'rb')))

    def background_pixels(self, r, cols_srgb, i, max_px=1500):
        """Composite every layer beneath text layer i inside its box at 256-px canvas scale."""
        layers = dict(self.t.get(r['id'], []))
        scale = 256.0 / max(r['cw'], r['ch'])
        W, H = int(round(r['cw'] * scale)), int(round(r['ch'] * scale))
        x, y, w, h = r['feat'][i, 0] - r['feat'][i, 2] / 2, r['feat'][i, 1] - r['feat'][i, 3] / 2, r['feat'][i, 2], r['feat'][i, 3]
        x0, y0 = max(0, int(x * W)), max(0, int(y * H))
        x1, y1 = min(W, max(x0 + 1, int(np.ceil((x + w) * W)))), min(H, max(y0 + 1, int(np.ceil((y + h) * H))))
        canvas = np.ones((y1 - y0, x1 - x0, 3), np.float32)
        for j in range(i):
            if j not in layers:
                continue
            im = layers[j].astype(np.float32) / 255.0
            jx, jy = r['feat'][j, 0] - r['feat'][j, 2] / 2, r['feat'][j, 1] - r['feat'][j, 3] / 2
            ox, oy = int(round(jx * W)), int(round(jy * H))
            hh, ww = im.shape[:2]
            ax0, ay0 = max(x0, ox), max(y0, oy)
            ax1, ay1 = min(x1, ox + ww), min(y1, oy + hh)
            if ax1 <= ax0 or ay1 <= ay0:
                continue
            patch = im[ay0 - oy:ay1 - oy, ax0 - ox:ax1 - ox]
            rgb, a = patch[..., :3], patch[..., 3:4] * r['opacity'][j]
            if r['rec'][j]:
                rgb = np.broadcast_to(cols_srgb[j], rgb.shape)
            region = canvas[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
            canvas[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = rgb * a + region * (1 - a)
        px = canvas.reshape(-1, 3)
        if len(px) > max_px:
            px = px[np.linspace(0, len(px) - 1, max_px).astype(int)]
        return px


def prepare_audit(r, cols, thumbs):
    """Per text layer: background pixels under every observer (independent of the text colour)."""
    srgb = to_srgb(cols)
    out = []
    for i, _ in text_pairs(r):
        px = thumbs.background_pixels(r, srgb, i)
        views = [view(px, M) for _, _, M in COND_FULL]
        out.append({'i': i, 'px': px, 'Yw': np.stack([lum_wcag(lin(v)) for v in views]),
                    'Ya': np.stack([apca_y(v) for v in views])})
    return out


def audit_fail_plate(r, i, text_srgb, px, plate_srgb, opacity, metric='wcag'):
    """Pixel audit of one text layer after compositing a uniform backing plate (scrim) of given opacity."""
    bg = opacity * plate_srgb[None] + (1 - opacity) * px
    thr = threshold(r, i, metric)
    worst = 0.0
    for _, _, M in COND_FULL:
        f = view(text_srgb[None], M)
        g = view(bg, M)
        if metric == 'wcag':
            yt, yb = lum_wcag(lin(f)), lum_wcag(lin(g))
            c = (np.maximum(yt, yb) + 0.05) / (np.minimum(yt, yb) + 0.05)
        else:
            c = apca_abs(apca_y(f), apca_y(g))
        worst = max(worst, float((c < thr).mean()))
    return worst > PIX_Q


def audit_fail(r, cols, audit, metric='wcag'):
    """Pixel-level failure indicator per text layer (worst case over 16 observers)."""
    srgb = to_srgb(cols)
    f = []
    for a in audit:
        i = a['i']
        thr = threshold(r, i, metric)
        fv = np.stack([view(srgb[i:i + 1], M)[0] for _, _, M in COND_FULL])
        if metric == 'wcag':
            yt = lum_wcag(lin(fv))[:, None]
            yb = a['Yw']
            c = (np.maximum(yt, yb) + 0.05) / (np.minimum(yt, yb) + 0.05)
        else:
            c = apca_abs(apca_y(fv)[:, None], a['Ya'])
        frac_bad = (c < thr).mean(1)
        f.append(bool((frac_bad > PIX_Q).any()))
    return np.array(f, bool)


def crc_lambda(losses, lams, alpha, B=1.0):
    """Conformal risk control: smallest lambda whose inflated empirical risk is <= alpha.

    losses: n x L matrix (designs x lambda grid), assumed non-increasing in lambda
    after the monotone envelope is applied."""
    env = np.maximum.accumulate(losses[:, ::-1], axis=1)[:, ::-1]
    n = len(env)
    risk = (n / (n + 1)) * env.mean(0) + B / (n + 1)
    ok = np.nonzero(risk <= alpha)[0]
    return (lams[ok[0]] if len(ok) else lams[-1]), env
