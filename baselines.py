"""Non-generative reference methods.

freq    : role- and size-conditioned colour frequency prior
knn     : retrieval of the nearest training template and colour transfer by layer matching
imgpal  : image-level palette regression from frozen photo features (ResNet-50, ViT-B/16 or CLIP),
          followed by area-ordered palette-to-layer assignment
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

import colorlib as cl

T_TEXT, T_IMG = 1, 2


def role_key(r, i):
    area = r['feat'][i, 2] * r['feat'][i, 3]
    sb = 0 if area < 0.02 else (1 if area < 0.2 else 2)
    return (int(r['type'][i]), sb, int(i == 0))


class FreqPrior:
    def __init__(self, train, K):
        self.K = K
        self.counts = {}
        for r in train:
            for i in np.nonzero(r['rec'])[0]:
                k = role_key(r, i)
                self.counts.setdefault(k, np.ones(K) * 0.1)
                self.counts[k][r['tok'][i]] += 1
        self.glob = sum(self.counts.values())

    def predict(self, r, lock, rng, greedy):
        tok = r['tok'].copy()
        for i in np.nonzero(r['rec'] & ~lock)[0]:
            p = self.counts.get(role_key(r, i), self.glob)
            tok[i] = int(np.argmax(p)) if greedy else int(rng.choice(self.K, p=p / p.sum()))
        return tok


def design_vector(r):
    types = np.bincount(r['type'], minlength=5) / max(r['n'], 1)
    fmt = np.zeros(67)
    fmt[r['format']] = 1
    return np.concatenate([np.asarray(r['title_emb'], np.float32), types * 2, fmt * 0.5])


class KNN:
    def __init__(self, train):
        self.train = train
        self.X = np.stack([design_vector(r) for r in train])
        self.X /= np.linalg.norm(self.X, axis=1, keepdims=True) + 1e-9

    def predict(self, r, lock, rng, greedy, cb):
        v = design_vector(r)
        v /= np.linalg.norm(v) + 1e-9
        sims = self.X @ v
        order = np.argsort(-sims)[:5]
        nb = self.train[order[0] if greedy else rng.choice(order)]
        cols = r['col'].copy()
        free = np.nonzero(r['rec'] & ~lock)[0]
        src = np.nonzero(nb['rec'])[0]
        if len(free) == 0 or len(src) == 0:
            return cols
        cost = np.zeros((len(free), len(src)))
        for a, i in enumerate(free):
            for c, j in enumerate(src):
                cost[a, c] = (3.0 * (r['type'][i] != nb['type'][j]) + np.abs(r['feat'][i, :4] - nb['feat'][j, :4]).sum()
                              + abs(r['feat'][i, 6] - nb['feat'][j, 6]))
        ri, ci = linear_sum_assignment(cost)
        for a, c in zip(ri, ci):
            cols[free[a]] = nb['col'][src[c]]
        unmatched = set(range(len(free))) - set(ri)
        for a in unmatched:
            cols[free[a]] = nb['col'][src[int(np.argmin(cost[a]))]]
        return cols


def design_palette(r, P=5):
    idx = np.nonzero(r['rec'])[0]
    c = r['col'][idx]
    w = np.sqrt(np.maximum(r['feat'][idx, 2] * r['feat'][idx, 3], 1e-4))
    if len(idx) <= P:
        pal = c[np.argsort(-w)]
        pal = np.concatenate([pal, np.repeat(pal[-1:], P - len(pal), 0)])
        return pal
    km = KMeans(P, n_init=3, random_state=0).fit(c, sample_weight=w)
    cw = np.bincount(km.labels_, weights=w, minlength=P)
    return km.cluster_centers_[np.argsort(-cw)]


class PaletteMLP(nn.Module):
    def __init__(self, din, P=5):
        super().__init__()
        self.P = P
        self.net = nn.Sequential(nn.Linear(din, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 3 * P))

    def forward(self, x):
        return self.net(x).view(-1, self.P, 3)


def photo_feature(r, emb, dim):
    vs = [emb['img'][(r['id'], i)] for i in range(r['n']) if r['type'][i] == T_IMG and (r['id'], i) in emb['img']]
    has = float(len(vs) > 0)
    v = np.mean(np.stack(vs).astype(np.float32), 0) if vs else np.zeros(dim, np.float32)
    fmt = np.zeros(67, np.float32)
    fmt[r['format']] = 1
    return np.concatenate([v, [has], fmt]).astype(np.float32)


def train_palette_regressor(train, emb, dim, epochs=30, seed=0):
    torch.manual_seed(seed)
    X = torch.tensor(np.stack([photo_feature(r, emb, dim) for r in train]))
    Y = torch.tensor(np.stack([design_palette(r) for r in train]), dtype=torch.float32)
    m = PaletteMLP(X.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for s in range(0, len(X), 256):
            ii = perm[s:s + 256]
            pred = m(X[ii])
            # permutation-invariant loss: Hungarian-free Chamfer distance in OKLab
            d = torch.cdist(pred, Y[ii])
            loss = d.min(1).values.mean() + d.min(2).values.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return m


def assign_palette(r, pal, lock):
    """Area-ordered assignment; text layers take the palette colour farthest (CIEDE2000) from their background."""
    cols = r['col'].copy()
    free = [i for i in np.nonzero(r['rec'] & ~lock)[0]]
    nontext = sorted([i for i in free if r['type'][i] != T_TEXT], key=lambda i: -r['feat'][i, 2] * r['feat'][i, 3])
    for k, i in enumerate(nontext):
        cols[i] = pal[min(k, len(pal) - 1)]
    lab = cl.srgb_to_cielab(cl.oklab_to_srgb(torch.tensor(pal, dtype=torch.float32)))
    for i in [i for i in free if r['type'][i] == T_TEXT]:
        j = r['bg'][i]
        if j >= 0:
            bgc = cols[j] if r['rec'][j] else r['fixed'][j, :3]
            bl = cl.srgb_to_cielab(cl.oklab_to_srgb(torch.tensor(bgc[None], dtype=torch.float32)))
            de = cl.ciede2000(lab, bl.expand_as(lab)).numpy()
            cols[i] = pal[int(np.argmax(de))]
        else:
            cols[i] = pal[-1]
    return cols
