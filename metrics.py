"""Evaluation metrics shared by every method."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import access as AC
import colorlib as cl
import harmony as hm

R_ON, R_UNDER, R_ADJ = 2, 3, 4


def lab(oklab):
    return cl.srgb_to_cielab(cl.oklab_to_srgb(torch.as_tensor(np.asarray(oklab), dtype=torch.float32))).numpy()


def colour_of(r, cols):
    """OKLab colour of every layer that has one (recolourable or fixed); NaN otherwise."""
    out = np.full((r['n'], 3), np.nan, np.float32)
    out[r['rec']] = cols[r['rec']]
    fx = ~r['rec'] & r['has_fixed']
    out[fx] = r['fixed'][fx, :3]
    return out


def fidelity(r, cols, lock):
    free = r['rec'] & ~lock
    if free.sum() == 0:
        return np.nan, np.nan
    de = cl.ciede2000(torch.tensor(lab(cols[free])), torch.tensor(lab(r['col'][free]))).numpy()
    return float(de.mean()), float((de <= 10).mean())


def pairs(r):
    out = []
    for i in range(r['n']):
        for j in range(i):
            if r['rel'][i, j] in (R_ON, R_UNDER, R_ADJ):
                out.append((i, j))
    return out


def ou_luo(r, cols):
    c = colour_of(r, cols)
    pp = [(i, j) for i, j in pairs(r) if not (np.isnan(c[i]).any() or np.isnan(c[j]).any())]
    if not pp:
        return np.nan
    L = lab(c)
    a = np.array([L[i] for i, _ in pp])
    b = np.array([L[j] for _, j in pp])
    return float(hm.ou_luo_harmony(a, b).mean())


def template_dev(r, cols):
    c = colour_of(r, cols)
    ok = ~np.isnan(c).any(1)
    if ok.sum() < 2:
        return np.nan
    area = r['feat'][:, 2] * r['feat'][:, 3]
    h, w = hm.hue_weights(c[ok], area[ok])
    return float(np.degrees(hm.fit_template(h, w)[2]))


def access_rates(r, cols, thumbs=None, audit=None):
    out = {}
    for m in ('wcag', 'apca'):
        pf = AC.proxy_fail(r, cols, m)
        out[f'proxy_{m}'] = pf
        if thumbs is not None:
            if audit is None:
                audit = AC.prepare_audit(r, cols, thumbs)
            out[f'pix_{m}'] = AC.audit_fail(r, cols, audit, m)
    return out, audit


# --- classifier two-sample test --------------------------------------------------

class Critic(nn.Module):
    def __init__(self, d=96):
        super().__init__()
        self.inp = nn.Linear(24 + 5 + 4, d)
        enc = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, 2)
        self.out = nn.Linear(d, 1)

    def forward(self, x, valid):
        h = self.enc(self.inp(x), src_key_padding_mask=~valid)
        h = (h * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        return self.out(h).squeeze(-1)


def critic_tensor(recs, colsets):
    N = max(r['n'] for r in recs)
    X = np.zeros((len(recs), N, 33), np.float32)
    V = np.zeros((len(recs), N), bool)
    for b, (r, cols) in enumerate(zip(recs, colsets)):
        c = colour_of(r, cols)
        has = ~np.isnan(c).any(1)
        X[b, :r['n'], :24] = r['feat']
        X[b, np.arange(r['n']), 24 + r['type']] = 1
        X[b, :r['n'], 29:32] = np.nan_to_num(c) * np.array([3, 10, 10])
        X[b, :r['n'], 32] = has
        V[b, :r['n']] = True
    return torch.tensor(X), torch.tensor(V)


def c2st_auc(recs, real, fake, folds=5, seed=0, epochs=10):
    """Held-out ROC-AUC of a critic separating designer colourings from generated ones.

    Folds are split by design so the same layout never appears in both training
    and evaluation folds. 0.5 means indistinguishable."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    ids = rng.permutation(len(recs))
    scores, labels = [], []
    for f in range(folds):
        te = ids[f::folds]
        tr = np.setdiff1d(ids, te)
        torch.manual_seed(seed + f)
        crit = Critic()
        opt = torch.optim.AdamW(crit.parameters(), lr=1e-3, weight_decay=1e-4)
        trr = [recs[i] for i in tr]
        Xr, Vr = critic_tensor(trr, [real[i] for i in tr])
        Xf, Vf = critic_tensor(trr, [fake[i] for i in tr])
        X = torch.cat([Xr, Xf])
        V = torch.cat([Vr, Vf])
        N = max(Xr.shape[1], Xf.shape[1])
        y = torch.cat([torch.ones(len(tr)), torch.zeros(len(tr))])
        for ep in range(epochs):
            perm = torch.randperm(len(X))
            for s in range(0, len(X), 128):
                ii = perm[s:s + 128]
                loss = F.binary_cross_entropy_with_logits(crit(X[ii], V[ii]), y[ii])
                opt.zero_grad()
                loss.backward()
                opt.step()
        crit.eval()
        ter = [recs[i] for i in te]
        with torch.no_grad():
            Xr, Vr = critic_tensor(ter, [real[i] for i in te])
            Xf, Vf = critic_tensor(ter, [fake[i] for i in te])
            scores += crit(Xr, Vr).tolist() + crit(Xf, Vf).tolist()
            labels += [1] * len(te) + [0] * len(te)
    return float(roc_auc_score(labels, scores))
