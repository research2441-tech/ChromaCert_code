"""Produce colour assignments for the test (or validation) split.

python run_eval.py --method full --seed 0 --protocol P|G [--split test] [--steps 8] [--gamma 1.0]
Protocol P: completion with 50% of recolourable layers locked (greedy decoding).
Protocol G: free generation with no locked layers (stochastic decoding).
Output: preds/<tag>.pkl  {design_id: n x 3 OKLab array}
"""
import argparse
import os
import pickle
import time

import numpy as np
import torch

import baselines as BL
import chroma as C
import harmony as hm
from model import ElementGraphNet

p = argparse.ArgumentParser()
p.add_argument('--method', required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--protocol', default='P')
p.add_argument('--split', default='test')
p.add_argument('--steps', type=int, default=8)
p.add_argument('--gamma', type=float, default=0.25)
p.add_argument('--lam', default='1,1,0.5')
p.add_argument('--lock', type=float, default=None)
p.add_argument('--ckpt', default=None)
p.add_argument('--tag', default=None)
p.add_argument('--limit', type=int, default=0)
p.add_argument('--threads', type=int, default=2)
a = p.parse_args()
torch.set_num_threads(a.threads)
os.makedirs('preds', exist_ok=True)

K = 256
lock_frac = a.lock if a.lock is not None else (0.5 if a.protocol == 'P' else 0.0)
greedy = a.protocol == 'P'
tag = a.tag or f'{a.method}_{a.protocol}_s{a.seed}' + ('' if a.split == 'test' else f'_{a.split}')


def load_net(path):
    ck = torch.load(path, weights_only=False)
    ar = ck['args']
    net = ElementGraphNet(ar['K'], d=ar['d'], layers=ar['layers'], heads=ar['heads'], use_rel=not ar['no_rel'],
                          use_sem=not ar['no_sem'], out='eps' if ar['objective'] == 'gauss' else 'tokens')
    net.load_state_dict(ck['state'])
    net.eval()
    return net, ar


ckpt = a.ckpt
if ckpt is None and a.method in ('mdm', 'mdm_guide', 'full', 'dpo', 'mcr', 'gauss'):
    base = {'mdm': 'mdm', 'mdm_guide': 'mdm', 'full': 'dpo', 'dpo': 'dpo', 'mcr': 'mlm', 'gauss': 'gauss'}[a.method]
    ckpt = f'runs/{base}_s{a.seed}.pt'
net, ar = load_net(ckpt) if ckpt else (None, None)
if ar is not None:
    K = ar['K']
cb = np.load(f'{C.DATA}/codebook_{K}.npy')
emb_path = f'{C.DATA}/emb_clip.pkl'
emb = pickle.load(open(emb_path, 'rb')) if (os.path.exists(emb_path) and (ar is None or not ar['no_sem'])) else None
st = C.Store(a.split, cb, emb)
recs = st.d[:a.limit] if a.limit else st.d
locks = C.design_locks(recs, lock_frac, 0)
out = {}
t0 = time.time()
rng = np.random.default_rng(a.seed)

if a.method == 'freq':
    tr = C.Store('train', cb)
    fp = BL.FreqPrior(tr.d, K)
    for r, l in zip(recs, locks):
        out[r['id']] = np.where(r['rec'][:, None], cb[fp.predict(r, l, rng, greedy).clip(0, K - 1)], r['col'])
elif a.method == 'knn':
    tr = C.Store('train', cb, emb)
    kn = BL.KNN(tr.d)
    for r, l in zip(recs, locks):
        out[r['id']] = kn.predict(r, l, rng, greedy, cb)
elif a.method.startswith('img_'):
    which = a.method[4:]
    e2 = emb if which == 'clip' else pickle.load(open(f'{C.DATA}/emb_{which}.pkl', 'rb'))
    dim = len(next(iter(e2['img'].values())))
    tr = C.Store('train', cb)
    m = BL.train_palette_regressor(tr.d, e2, dim, seed=a.seed)
    with torch.no_grad():
        for r, l in zip(recs, locks):
            pal = m(torch.tensor(BL.photo_feature(r, e2, dim))[None])[0].numpy()
            if not greedy:
                pal = pal + rng.normal(0, 0.02, pal.shape)
            out[r['id']] = BL.assign_palette(r, pal, l)
elif a.method == 'gauss':
    for s in range(0, len(recs), 64):
        rr, ll = recs[s:s + 64], locks[s:s + 64]
        res = C.sample_gauss(net, st, rr, ll, seed=a.seed * 1000 + s)
        for r, l, c in zip(rr, ll, res):
            out[r['id']] = np.where((r['rec'] & ~l)[:, None], c, r['col'])
else:
    guide = None
    if a.method in ('mdm_guide', 'full'):
        tr = C.Store('train', cb)
        prior_path = f'{C.DATA}/lightness_prior.pkl'
        if not os.path.exists(prior_path):
            pickle.dump(hm.LightnessPrior.fit(tr.d).table, open(prior_path, 'wb'))
        lh, ls, ll_ = [float(x) for x in a.lam.split(',')]
        guide = C.Guide(cb, hm.LightnessPrior(pickle.load(open(prior_path, 'rb'))), lh, ls, ll_)
    steps = 1 if a.method == 'mcr' else a.steps
    for s in range(0, len(recs), 64):
        rr, lk = recs[s:s + 64], locks[s:s + 64]
        toks = C.sample(net, st, rr, lk, steps=steps, guide=guide, gamma=a.gamma, seed=a.seed * 1000 + s, greedy=greedy)
        for r, tk in zip(rr, toks):
            out[r['id']] = np.where(r['rec'][:, None], cb[np.clip(tk, 0, K - 1)], r['col'])
elapsed = time.time() - t0
pickle.dump({'pred': out, 'locks': {r['id']: l for r, l in zip(recs, locks)}, 'seconds': elapsed,
             'per_design_ms': 1000 * elapsed / len(recs)}, open(f'preds/{tag}.pkl', 'wb'))
print(tag, f'{1000 * elapsed / len(recs):.1f} ms/design')
