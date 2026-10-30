"""Aggregate per-design results into the numbers used by the tables and figures.

Writes res/summary.json. Requires res/<tag>.pkl from evaluate.py and res/crc_*.pkl from run_crc.py.
"""
import glob
import json
import os
import pickle

import numpy as np

import stats as S

R = 'res'


def load(tag):
    p = f'{R}/{tag}.pkl'
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else None


def scal(res, key):
    if res is None or key not in res:
        return np.nan
    if key == 'c2st':
        return res['c2st']
    if key == 'ms':
        return res['ms'] or np.nan
    v = S.per_design(res, key)
    if key.startswith('proxy') or key.startswith('pix'):
        # rate over text layers (pooled), not mean of design means
        allv = np.concatenate([np.asarray(x, float) for x in res[key] if len(x)])
        return float(allv.mean())
    return float(np.nanmean(v))


KEYS = ['de', 'hit10', 'ou', 'proxy_wcag', 'proxy_apca', 'pix_wcag', 'pix_apca', 'c2st', 'ms']


def method_row(name, seeds=(0, 1, 2), prefix=None):
    out = {}
    for proto in ('P', 'G'):
        vals = {k: [] for k in KEYS}
        for s in seeds:
            tag = (prefix or f'{name}_{proto}_s{{s}}').format(s=s) if prefix else f'{name}_{proto}_s{s}'
            r = load(tag)
            if r is None:
                continue
            for k in KEYS:
                vals[k].append(scal(r, k))
        for k, v in vals.items():
            v = [x for x in v if not np.isnan(x)]
            if v:
                out[f'{proto}_{k}'] = (float(np.mean(v)), float(np.std(v)) if len(v) > 1 else None, len(v))
    return out


def tests(ref_tag, others, key):
    ref = load(ref_tag)
    rows = []
    for name, tag in others:
        o = load(tag)
        if o is None or ref is None:
            continue
        ids = [i for i in ref['ids'] if i in set(o['ids'])]
        ri = {i: k for k, i in enumerate(ref['ids'])}
        oi = {i: k for k, i in enumerate(o['ids'])}
        a = S.per_design(ref, key)[[ri[i] for i in ids]]
        b = S.per_design(o, key)[[oi[i] for i in ids]]
        a, b = S.paired(a, b)
        mean, lo, hi = S.bootstrap_ci(a, b)
        rows.append({'name': name, 'n': int(len(a)), 'diff': mean, 'lo': lo, 'hi': hi, 'p': S.wilcoxon_p(a, b),
                     'rrb': S.rank_biserial(a, b)})
    q = S.bh([r['p'] for r in rows]) if rows else []
    for r, qq in zip(rows, q):
        r['q'] = float(qq)
    return rows


def crc_summary(path, alphas=(0.05, 0.10, 0.15, 0.20), resplits=200, seed=0):
    import access as AC
    D = pickle.load(open(path, 'rb'))
    taus = D['taus']
    cal, test = D['cal'], D['test']
    path_n = 1 + len(taus)
    P = np.arange(path_n)
    out = {'n_cal': len(cal), 'n_test': len(test), 'taus': taus.tolist(), 'lams_margin': D['lams'].tolist(),
           'none_test': float(np.mean([d['none'].mean() for d in test])),
           'none_layer': float(np.mean(np.concatenate([d['none'] for d in test]))),
           'proxy0_test': float(np.mean(np.concatenate([d['proxy0'] for d in test]))),
           'nominal_layer': float(np.mean(np.concatenate([d['fail'][0] for d in test]))),
           'nominal_edit': float(np.mean(np.concatenate([d['edited'][0] for d in test]))),
           'nominal_cost_edited': float(np.mean(np.concatenate([d['cost'][0][d['edited'][0]] for d in test]))),
           'margin_curve_test': np.mean(np.stack([d['loss_margin'] for d in test]), 0).tolist(),
           'margin_edit_test': np.mean(np.stack([d['edited'].mean(1) for d in test]), 0).tolist(),
           'margin_cost_test': np.mean(np.stack([d['cost'].mean(1) for d in test]), 0).tolist(),
           'scrim_cost_edited': float(np.mean(np.concatenate([d['scrim_cost'] for d in test]))),
           'policies': {}}
    for pol, lkey, dkey in (('defer', 'loss', 'defer'), ('scrim', 'loss_plate', 'scrim')):
        Lc = np.stack([d[lkey] for d in cal])
        Lt = np.stack([d[lkey] for d in test])
        Dt = np.stack([d[dkey] for d in test])
        pooled = np.concatenate([Lc, Lt])
        rng = np.random.default_rng(seed)
        res = {'curve_cal': Lc.mean(0).tolist(), 'curve_test': Lt.mean(0).tolist(), 'esc_test': Dt.mean(0).tolist(),
               'alphas': {}}
        for a in alphas:
            k, _ = AC.crc_lambda(Lc, P, a)
            k = int(k)
            rs = []
            for _ in range(resplits):
                perm = rng.permutation(len(pooled))
                h = len(pooled) // 2
                kk, _ = AC.crc_lambda(pooled[perm[:h]], P, a)
                rs.append(float(pooled[perm[h:], int(kk)].mean()))
            res['alphas'][str(a)] = {'k': k, 'lam': (0.0 if k == 0 else float(taus[k - 1])),
                                     'risk': float(Lt[:, k].mean()), 'esc': float(Dt[:, k].mean()),
                                     'resplit_mean': float(np.mean(rs)), 'resplit_q95': float(np.percentile(rs, 95)),
                                     'resplit_viol': float(np.mean(np.array(rs) > a)), 'resplit_risks': rs}
        k10 = res['alphas']['0.1']['k']
        grp = {}
        for d in test:
            grp.setdefault(int(d['group']), []).append(d[lkey][k10])
        res['group_risk'] = {g: (float(np.mean(v)), len(v)) for g, v in grp.items()}
        # oracle: smallest escalated fraction that reaches each alpha if failing layers were known
        allf = np.concatenate([d['fail'][0] for d in test])
        res['oracle_esc'] = {str(a): float(max(0.0, allf.mean() - a)) for a in alphas}
        out['policies'][pol] = res
    return out


def group_de(tag='full_P_s0'):
    import chroma as C
    st = C.Store('test', np.load(f'{C.DATA}/codebook_256.npy'))
    g = {r['id']: r['group'] for r in st.d}
    r = load(tag)
    out = {}
    for i, v in zip(r['ids'], r['de']):
        if not np.isnan(v):
            out.setdefault(str(g[i]), []).append(v)
    json.dump({k: float(np.mean(v)) for k, v in out.items()}, open('res/group_de.json', 'w'), indent=1)


if __name__ == '__main__':
    summ = {}
    rows = {}
    for name in ('freq', 'knn', 'img_resnet50', 'img_vitb16', 'img_clip'):
        rows[name] = method_row(name, seeds=(0,))
    for name in ('mcr', 'gauss', 'mdm', 'mdm_guide', 'dpo', 'full'):
        rows[name] = method_row(name)
    rows['mdm_guide_s0'] = method_row('mdm_guide', seeds=(0,))
    rows['designer'] = method_row('designer', seeds=(0,))
    for ab in ('norel', 'nosem', 'K128', 'K512', 'nohue', 'nosep', 'nolig'):
        rows[f'abl_{ab}'] = method_row(f'abl_{ab}', seeds=(0,))
    for g in (0.5, 1.0, 2.0, 4.0):
        rows[f'sens_gamma{g}'] = method_row(f'sens_gamma{g}', seeds=(0,))
    for st in (1, 2, 4, 16):
        rows[f'sens_steps{st}'] = method_row(f'sens_steps{st}', seeds=(0,))
    for lf in (0.25, 0.75):
        rows[f'sens_lock{lf}'] = method_row(f'sens_lock{lf}', seeds=(0,))
        rows[f'sens_lock{lf}_mcr'] = method_row(f'sens_lock{lf}_mcr', seeds=(0,))
    summ['rows'] = rows
    tm = {}
    for kind in ('mlm', 'gauss', 'mdm', 'dpo'):
        js = [json.load(open(f)) for f in sorted(glob.glob(f'runs/{kind}_s[0-9].json'))]
        if kind == 'dpo':
            tm[kind] = {'min': float(np.mean([(j['pair_seconds'] + j['dpo_seconds']) / 60 for j in js]))}
        else:
            tm[kind] = {'params': js[0]['params'], 'min': float(np.mean([j['train_seconds'] / 60 for j in js]))}
    json.dump(tm, open('res/train_meta.json', 'w'), indent=1)
    base = [('Frequency prior', 'freq_P_s0'), ('Retrieval', 'knn_P_s0'), ('ResNet-50 palette', 'img_resnet50_P_s0'),
            ('ViT-B/16 palette', 'img_vitb16_P_s0'), ('CLIP palette', 'img_clip_P_s0'), ('One-shot MCM', 'mcr_P_s0'),
            ('Gaussian diffusion', 'gauss_P_s0')]
    summ['tests_de'] = tests('full_P_s0', base, 'de')
    summ['tests_pix'] = tests('full_G_s0', [(n, t.replace('_P_', '_G_')) for n, t in base], 'pix_wcag')
    for f in sorted(glob.glob(f'{R}/crc_*.pkl')):
        try:
            summ[os.path.basename(f)[:-4]] = crc_summary(f)
        except KeyError as e:
            print('skip (old format)', f, e)
    json.dump(summ, open(f'{R}/summary.json', 'w'), indent=1, default=float)
    group_de()
    print(json.dumps({k: v for k, v in rows.items() if k in ('freq', 'mcr', 'mdm', 'full')}, indent=0)[:3000])


