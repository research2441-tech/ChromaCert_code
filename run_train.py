"""Train one model variant.

python run_train.py --name full_s0 --objective mdm --seed 0 [--no-rel] [--no-sem] [--K 256]
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch

import chroma as C
from model import ElementGraphNet

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--objective', default='mdm', choices=['mdm', 'mlm', 'gauss'])
p.add_argument('--seed', type=int, default=0)
p.add_argument('--K', type=int, default=256)
p.add_argument('--epochs', type=int, default=20)
p.add_argument('--d', type=int, default=128)
p.add_argument('--layers', type=int, default=4)
p.add_argument('--heads', type=int, default=4)
p.add_argument('--no-rel', action='store_true')
p.add_argument('--no-sem', action='store_true')
p.add_argument('--threads', type=int, default=2)
a = p.parse_args()

torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
os.makedirs('runs', exist_ok=True)
cb = np.load(f'{C.DATA}/codebook_{a.K}.npy')
emb = None if a.no_sem else pickle.load(open(f'{C.DATA}/emb_clip.pkl', 'rb'))
st = C.Store('train', cb, emb)
net = ElementGraphNet(a.K, d=a.d, layers=a.layers, heads=a.heads, use_rel=not a.no_rel,
                      use_sem=not a.no_sem, out='eps' if a.objective == 'gauss' else 'tokens')
logf = open(f'runs/{a.name}.log', 'w')


def log(m):
    print(m, flush=True)
    logf.write(m + '\n')
    logf.flush()


t0 = time.time()
hist = C.train(net, st, a.epochs, objective=a.objective, seed=a.seed, log=log)
torch.save({'state': net.state_dict(), 'args': vars(a)}, f'runs/{a.name}.pt')
json.dump({'loss': hist, 'train_seconds': time.time() - t0,
           'params': sum(p.numel() for p in net.parameters())}, open(f'runs/{a.name}.json', 'w'))
log(f'done {time.time() - t0:.0f}s')
