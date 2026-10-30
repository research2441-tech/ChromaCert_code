"""Turn extracted Crello records into padded element-graph tensors.

Outputs (in data/):
  designs_<split>.pkl  : list of per-design dicts with numpy arrays
  codebook.npy         : K x 3 OKLab colour vocabulary fitted on training colours
  meta.json            : counts and filtering statistics
"""
import glob
import json
import math
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, os.path.dirname(__file__))
import colorlib as cl

MAXE = 40
T_SVG, T_TEXT, T_IMG, T_BG, T_MASK = 0, 1, 2, 3, 4
# relation ids used by the attention bias
R_NONE, R_SELF, R_ON, R_UNDER, R_ADJ, R_SAMEROLE = 0, 1, 2, 3, 4, 5
NREL = 6


def oklab(rgb01):
    return cl.srgb_to_oklab(torch.tensor(rgb01, dtype=torch.float32)).numpy()


def inter(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def gap(a, b):
    dx = max(0.0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
    return math.hypot(dx, dy)


def encode(d):
    els = d['elements']
    if len(els) > MAXE:
        return None
    n = len(els)
    rec = np.array([e['recolorable'] for e in els], bool)
    if rec.sum() < 2:
        return None
    feat = np.zeros((n, 24), np.float32)
    fixed = np.zeros((n, 9), np.float32)       # mean, q10, q90 OKLab of fixed layers
    has_fixed = np.zeros(n, bool)
    col = np.zeros((n, 3), np.float32)          # designer OKLab colour of recolourable layers
    rgb = np.zeros((n, 3), np.float32)
    for i, e in enumerate(els):
        x, y, w, h = e['box']
        cx, cy = x + w / 2, y + h / 2
        a = math.radians(e['angle'])
        feat[i, :12] = [cx, cy, w, h, math.log(max(w * h, 1e-6)), math.log(max(w, 1e-4) / max(h, 1e-4)),
                        i / max(n - 1, 1), e['opacity'], math.sin(a), math.cos(a), e['font_size'] * 10,
                        math.log1p(len(e['text']))]
        feat[i, 12 + e['type']] = 1.0
        feat[i, 17] = float(e['recolorable'])
        feat[i, 18] = min(e['palette_len'], 5) / 5.0
        feat[i, 19] = e.get('bold', 0.0)
        feat[i, 20] = float(i == 0)
        feat[i, 21] = float(w * h > 0.6)
        feat[i, 22] = float(e['type'] == T_TEXT and e['font_size'] * d['ch'] >= 24)
        feat[i, 23] = e.get('n_text_colors', 1) / 5.0
        if e['recolorable']:
            rgb[i] = np.array(e['color'], np.float32) / 255.0
            col[i] = oklab(rgb[i])
        elif e['summary'] is not None:
            s = e['summary']
            fixed[i] = np.concatenate([oklab(s['mean']), oklab(s['q10']), oklab(s['q90'])])
            has_fixed[i] = True
    rel = np.zeros((n, n), np.int8)
    ovl = np.zeros((n, n), np.float16)
    boxes = [e['box'] for e in els]
    for i in range(n):
        ai = max(boxes[i][2] * boxes[i][3], 1e-6)
        for j in range(n):
            if i == j:
                rel[i, j] = R_SELF
                continue
            it = inter(boxes[i], boxes[j])
            frac = it / ai
            ovl[i, j] = frac
            if frac > 0.02:
                rel[i, j] = R_ON if j < i else R_UNDER     # i lies on top of j / i lies under j
            elif gap(boxes[i], boxes[j]) < 0.03:
                rel[i, j] = R_ADJ
            elif els[i]['type'] == els[j]['type']:
                rel[i, j] = R_SAMEROLE
    # proxy background of each text layer: top-most layer beneath covering >=50% of its box
    bg = -np.ones(n, np.int64)
    for i, e in enumerate(els):
        if e['type'] != T_TEXT:
            continue
        for j in range(i - 1, -1, -1):
            if els[j]['type'] == T_TEXT:
                continue
            if ovl[i, j] >= 0.5 and els[j]['opacity'] >= 0.5 and (els[j]['recolorable'] or has_fixed[j]):
                bg[i] = j
                break
    return {'id': d['id'], 'group': d['group'], 'format': d['format'], 'category': d['category'],
            'cw': d['cw'], 'ch': d['ch'], 'n': n, 'type': np.array([e['type'] for e in els], np.int64),
            'feat': feat, 'fixed': fixed, 'has_fixed': has_fixed, 'rec': rec, 'col': col, 'rgb': rgb,
            'rel': rel, 'ovl': ovl, 'bg': bg, 'opacity': np.array([e['opacity'] for e in els], np.float32),
            'title': d['title'], 'keywords': d['keywords'],
            'texts': [e['text'] for e in els], 'font_px': np.array([e['font_size'] * d['ch'] for e in els], np.float32)}


def main(prep='prep', out='data', K=256):
    os.makedirs(out, exist_ok=True)
    meta = {}
    allc = []
    for split in ('train', 'validation', 'test'):
        recs, dropped = [], 0
        for f in sorted(glob.glob(f'{prep}/{split}-*_designs.pkl')):
            for d in pickle.load(open(f, 'rb')):
                r = encode(d)
                if r is None:
                    dropped += 1
                    continue
                recs.append(r)
        meta[split] = {'kept': len(recs), 'dropped': dropped}
        pickle.dump(recs, open(f'{out}/designs_{split}.pkl', 'wb'))
        if split == 'train':
            for r in recs:
                allc.append(r['col'][r['rec']])
        print(split, meta[split])
    allc = np.concatenate(allc)
    km = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=8192, n_init=3).fit(allc)
    np.save(f'{out}/codebook_{K}.npy', km.cluster_centers_.astype(np.float32))
    meta['n_train_colours'] = int(len(allc))
    json.dump(meta, open(f'{out}/meta.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
