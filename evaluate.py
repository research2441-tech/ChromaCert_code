"""Score every prediction file and store per-design metrics.

python evaluate.py preds/<tag>.pkl [--c2st] [--audit]
Writes res/<tag>.pkl with per-design arrays used by the statistics scripts.
"""
import argparse
import os
import pickle
import warnings

import numpy as np

warnings.filterwarnings('ignore')
import access as AC
import chroma as C
import metrics as M

p = argparse.ArgumentParser()
p.add_argument('files', nargs='+')
p.add_argument('--c2st', action='store_true')
p.add_argument('--audit', action='store_true')
p.add_argument('--split', default='test')
p.add_argument('--proxy-only', action='store_true')
a = p.parse_args()
os.makedirs('res', exist_ok=True)
cb = np.load(f'{C.DATA}/codebook_256.npy')
st = C.Store(a.split, cb)
byid = {r['id']: r for r in st.d}
thumbs = AC.Thumbs('prep', (a.split if a.split != 'test' else 'test',)) if a.audit else None

for f in a.files:
    P = pickle.load(open(f, 'rb'))
    tag = os.path.basename(f)[:-4]
    if a.proxy_only:
        old = pickle.load(open(f'res/{tag}.pkl', 'rb'))
        recs = [byid[i] for i in old['ids']]
        for m in ('wcag', 'apca'):
            old[f'proxy_{m}'] = [AC.proxy_fail(r, P['pred'][r['id']], m) for r in recs]
        pickle.dump(old, open(f'res/{tag}.pkl', 'wb'))
        print(tag, 'proxy', np.mean(np.concatenate(old['proxy_wcag'])), flush=True)
        continue
    ids = [i for i in P['pred'] if i in byid]
    recs = [byid[i] for i in ids]
    out = {'ids': ids, 'ms': P.get('per_design_ms')}
    de, hit, ou, td = [], [], [], []
    acc = {k: [] for k in ('proxy_wcag', 'proxy_apca', 'pix_wcag', 'pix_apca')}
    for r in recs:
        c = P['pred'][r['id']]
        lk = P['locks'][r['id']]
        d, h = M.fidelity(r, c, lk)
        de.append(d)
        hit.append(h)
        ou.append(M.ou_luo(r, c))
        td.append(M.template_dev(r, c))
        ar, _ = M.access_rates(r, c, thumbs)
        for k, v in ar.items():
            acc[k].append(v)
    out.update({'de': np.array(de), 'hit10': np.array(hit), 'ou': np.array(ou), 'tdev': np.array(td)})
    for k, v in acc.items():
        if v:
            out[k] = v
    if a.c2st:
        out['c2st'] = M.c2st_auc(recs, [r['col'] for r in recs], [P['pred'][r['id']] for r in recs])
    pickle.dump(out, open(f'res/{tag}.pkl', 'wb'))
    msg = f'{tag}: dE {np.nanmean(de):.2f} hit10 {np.nanmean(hit):.3f} ou {np.nanmean(ou):.3f} tdev {np.nanmean(td):.2f}'
    for k, v in acc.items():
        if v:
            msg += f' {k} {np.mean(np.concatenate(v)) if len(v) else np.nan:.3f}'
    if a.c2st:
        msg += f' c2st {out["c2st"]:.3f}'
    print(msg, flush=True)
