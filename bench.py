"""Clean latency benchmark (single thread, nothing else running): 300 test designs, protocol G.
Writes res/bench.json with ms per design for each sampler and for certification."""
import json
import pickle
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')
import access as AC
import chroma as C
import harmony as hm
from model import ElementGraphNet

torch.set_num_threads(1)
cb = np.load(f'{C.DATA}/codebook_256.npy')
emb = pickle.load(open(f'{C.DATA}/emb_clip.pkl', 'rb'))
st = C.Store('test', cb, emb)
recs = st.d[:300]
locks = C.design_locks(recs, 0.0, 0)


def load(name):
    ck = torch.load(f'runs/{name}.pt', weights_only=False)
    a = ck['args']
    net = ElementGraphNet(a['K'], d=a['d'], layers=a['layers'], heads=a['heads'], use_rel=not a['no_rel'],
                          use_sem=not a['no_sem'], out='eps' if a['objective'] == 'gauss' else 'tokens')
    net.load_state_dict(ck['state'])
    return net.eval()


guide = C.Guide(cb, hm.LightnessPrior(pickle.load(open(f'{C.DATA}/lightness_prior.pkl', 'rb'))))
out = {}
for name, fn in [('mcr_one_shot', lambda: [C.sample(load('mlm_s0'), st, recs[s:s + 64], locks[s:s + 64], steps=1) for s in range(0, 300, 64)]),
                 ('mdm_S8', lambda: [C.sample(load('mdm_s0'), st, recs[s:s + 64], locks[s:s + 64], steps=8) for s in range(0, 300, 64)]),
                 ('chromacert_S8_guided', lambda: [C.sample(load('dpo_s0'), st, recs[s:s + 64], locks[s:s + 64], steps=8, guide=guide, gamma=0.25) for s in range(0, 300, 64)]),
                 *[(f'chromacert_S{S_}_guided', (lambda S_=S_: [C.sample(load('dpo_s0'), st, recs[s:s + 64], locks[s:s + 64], steps=S_, guide=guide, gamma=0.25) for s in range(0, 300, 64)])) for S_ in (1, 2, 4, 16)],
                 ('gauss_ddim50', lambda: [C.sample_gauss(load('gauss_s0'), st, recs[s:s + 64], locks[s:s + 64]) for s in range(0, 300, 64)])]:
    t0 = time.time()
    fn()
    out[name] = 1000 * (time.time() - t0) / len(recs)
    print(name, out[name], flush=True)
# certification cost: nominal hue-preserving repair (margin 0) and evidence score for every auditable text layer
P = pickle.load(open('preds/full_G_s0.pkl', 'rb'))
t0 = time.time()
for r in recs:
    rc, _ = AC.repair(r, P['pred'][r['id']], 0.0, 'wcag')
    for i, j in AC.text_pairs(r):
        AC.proxy_min_contrast(r, rc, AC.to_srgb(rc[i]), j, 'wcag', i)
out['repair_score'] = 1000 * (time.time() - t0) / len(recs)
print('repair+score', out['repair_score'])
json.dump(out, open('res/bench.json', 'w'), indent=1)
