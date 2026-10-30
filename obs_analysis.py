"""Per-observer pixel-level failure rates (WCAG) for a prediction file, before and
after repair at a given margin. Writes res/obs_<tag>.json.

python obs_analysis.py preds/<tag>.pkl <margin> [--designer]
"""
import json
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
import access as AC
import chroma as C


def per_observer(r, cols, audit, metric='wcag'):
    srgb = AC.to_srgb(cols)
    rows = []
    for a in audit:
        i = a['i']
        thr = AC.threshold(r, i, metric)
        fv = np.stack([AC.view(srgb[i:i + 1], M)[0] for _, _, M in AC.COND_FULL])
        yt = AC.lum_wcag(AC.lin(fv))[:, None]
        c = (np.maximum(yt, a['Yw']) + 0.05) / (np.minimum(yt, a['Yw']) + 0.05)
        rows.append(((c < thr).mean(1) > AC.PIX_Q))
    return rows


def main(path, margin, designer=False):
    P = pickle.load(open(path, 'rb'))
    st = C.Store('test', np.load(f'{C.DATA}/codebook_256.npy'))
    th = AC.Thumbs('prep', ('test',))
    gen, rep, des = [], [], []
    for r in st.d:
        if r['id'] not in P['pred'] or not AC.text_pairs(r):
            continue
        cols = P['pred'][r['id']]
        audit = AC.prepare_audit(r, cols, th)
        gen += per_observer(r, cols, audit)
        rc, _ = AC.repair(r, cols, margin, 'wcag')
        rep += per_observer(r, rc, audit)
        if designer:
            ad = AC.prepare_audit(r, r['col'], th)
            des += per_observer(r, r['col'], ad)
    names = [f'{k}-{s:.1f}' if k != 'normal' else 'normal' for k, s, _ in AC.COND_FULL]
    out = {'observers': names, 'generated': np.mean(gen, 0).tolist(), 'repaired': np.mean(rep, 0).tolist(),
           'n_text': len(gen), 'margin': margin}
    if designer:
        out['designer'] = np.mean(des, 0).tolist()
    tag = path.split('/')[-1][:-4]
    json.dump(out, open(f'res/obs_{tag}.json', 'w'), indent=1)
    print(json.dumps(out)[:600])


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]), '--designer' in sys.argv)
