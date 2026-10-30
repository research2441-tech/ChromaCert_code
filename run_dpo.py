"""Masked-diffusion preference optimisation on designer-versus-model pairs.

python run_dpo.py --ref runs/mdm_s0.pt --name dpo_s0 --seed 0 [--pairs 6000] [--beta 0.1]
"""
import argparse
import json
import pickle
import time

import numpy as np
import torch

import chroma as C
import colorlib as cl
from model import ElementGraphNet

p = argparse.ArgumentParser()
p.add_argument('--ref', required=True)
p.add_argument('--name', required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--pairs', type=int, default=6000)
p.add_argument('--beta', type=float, default=0.1)
p.add_argument('--epochs', type=int, default=2)
p.add_argument('--min_de', type=float, default=5.0)
p.add_argument('--threads', type=int, default=2)
a = p.parse_args()
torch.set_num_threads(a.threads)

ck = torch.load(a.ref, weights_only=False)
ar = ck['args']


def make():
    net = ElementGraphNet(ar['K'], d=ar['d'], layers=ar['layers'], heads=ar['heads'], use_rel=not ar['no_rel'], use_sem=not ar['no_sem'])
    net.load_state_dict(ck['state'])
    return net


ref, pol = make(), make()
cb = np.load(f'{C.DATA}/codebook_{ar["K"]}.npy')
emb = None if ar['no_sem'] else pickle.load(open(f'{C.DATA}/emb_clip.pkl', 'rb'))
st = C.Store('train', cb, emb)
rng = np.random.default_rng(a.seed)
idx = rng.choice(len(st.d), min(a.pairs, len(st.d)), replace=False)
pairs, t0 = [], time.time()
ref.eval()
for s in range(0, len(idx), 64):
    ii = idx[s:s + 64]
    recs = [st.d[i] for i in ii]
    fr = float(rng.choice([0.0, 0.25, 0.5]))
    locks = C.design_locks(recs, fr, a.seed + 7)
    toks = C.sample(ref, st, recs, locks, steps=8, seed=a.seed * 7919 + s)
    for i, r, l, tk in zip(ii, recs, locks, toks):
        free = r['rec'] & ~l
        if free.sum() == 0:
            continue
        lose = np.where(r['rec'], tk, r['tok'])
        a_lab = cl.srgb_to_cielab(cl.oklab_to_srgb(torch.tensor(cb[np.clip(lose[free], 0, len(cb) - 1)])))
        b_lab = cl.srgb_to_cielab(cl.oklab_to_srgb(torch.tensor(r['col'][free])))
        de = cl.ciede2000(a_lab, b_lab).mean().item()
        if de < a.min_de:
            continue
        pairs.append({'idx': int(i), 'lock': l, 'lose': lose.astype(np.int64), 'de': de})
print(f'{len(pairs)} pairs in {time.time() - t0:.0f}s', flush=True)
logf = open(f'runs/{a.name}.log', 'w')


def log(m):
    print(m, flush=True)
    logf.write(m + '\n')


t1 = time.time()
hist = C.dpo_finetune(pol, ref, st, pairs, epochs=a.epochs, beta=a.beta, seed=a.seed, log=log)
args = dict(ar)
args.update({'dpo_beta': a.beta, 'dpo_pairs': len(pairs), 'ref': a.ref})
torch.save({'state': pol.state_dict(), 'args': args}, f'runs/{a.name}.pt')
json.dump({'hist': hist, 'pairs': len(pairs), 'pair_seconds': t1 - t0, 'dpo_seconds': time.time() - t1},
          open(f'runs/{a.name}.json', 'w'))
