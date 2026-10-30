"""Classifier two-sample test for many generation files with one fixed protocol:
1,000 test designs drawn once with seed 0, five folds split by design, eight epochs.

python c2st_all.py preds/<tag1>.pkl preds/<tag2>.pkl ...   (updates res/<tag>.pkl['c2st'])
"""
import os
import pickle
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')
import chroma as C
import metrics as M

torch.set_num_threads(int(os.environ.get('NT', '1')))
st = C.Store('test', np.load(f'{C.DATA}/codebook_256.npy'))
rng = np.random.default_rng(0)
idx = sorted(rng.choice(len(st.d), 1000, replace=False).tolist())
recs = [st.d[i] for i in idx]
for f in sys.argv[1:]:
    tag = os.path.basename(f)[:-4]
    P = pickle.load(open(f, 'rb'))
    sub = [r for r in recs if r['id'] in P['pred']]
    auc = M.c2st_auc(sub, [r['col'] for r in sub], [P['pred'][r['id']] for r in sub], folds=5, seed=0, epochs=8)
    rp = f'res/{tag}.pkl'
    res = pickle.load(open(rp, 'rb')) if os.path.exists(rp) else {'ids': [r['id'] for r in sub]}
    res['c2st'] = auc
    res['c2st_n'] = len(sub)
    pickle.dump(res, open(rp, 'wb'))
    print(tag, f'c2st {auc:.3f} n={len(sub)}', flush=True)
