"""Accessibility certification experiments.

python run_crc.py --cal preds/<tag>_validation.pkl --test preds/<tag>.pkl --metric wcag --out res/crc_<tag>_wcag.pkl
For every design with at least one auditable text layer, the script computes the
per-design pixel-level failure fraction as a function of the repair margin lambda,
then calibrates lambda by conformal risk control.
"""
import argparse
import pickle
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')
import access as AC
import chroma as C
import colorlib as cl

p = argparse.ArgumentParser()
p.add_argument('--cal', required=True)
p.add_argument('--test', required=True)
p.add_argument('--metric', default='wcag')
p.add_argument('--out', required=True)
a = p.parse_args()
LAMS = np.round(np.arange(0.0, 1.0001, 0.05), 3)
TAUS = np.round(np.arange(0.05, 2.0001, 0.05), 3)  # thresholds on the evidence contrast score s_i (escalating)
NPATH = 1 + len(TAUS)                      # escalation path: nominal repair, then deferral by score
OPAC = np.round(np.arange(0.1, 1.0001, 0.1), 2)  # scrim policy: margin first, then backing-plate opacity
OPAC_MAIN = 0.7                            # scrim opacity used on the main escalation path
NPLATE = 1 + len(TAUS)
WHITE, BLACK = np.ones(3), np.zeros(3)


def candidates(r, cols, i, j, metric):
    """Hue-preserving OKLCH candidates for text layer i with proxy contrast and CIEDE2000 cost."""
    lch = cl.oklab_to_oklch(torch.tensor(cols[i], dtype=torch.float32)).numpy()
    labs = []
    for scale in (1.0, 0.75, 0.5, 0.25, 0.0):
        lab = cl.oklch_to_oklab(torch.tensor(np.stack([AC._L, np.full_like(AC._L, lch[1] * scale), np.full_like(AC._L, lch[2])], 1), dtype=torch.float32))
        ok = cl.oklab_in_gamut(lab).numpy()
        labs.append(lab.numpy()[ok])
    labs = np.concatenate([cols[i:i + 1]] + labs)          # row 0 = unrepaired colour
    rgbs = AC.to_srgb(labs)
    cmin = AC.proxy_min_contrast(r, cols, rgbs, j, metric)
    o = cl.srgb_to_cielab(torch.tensor(rgbs[:1], dtype=torch.float32))
    de = cl.ciede2000(cl.srgb_to_cielab(torch.tensor(rgbs, dtype=torch.float32)), o.expand(len(rgbs), 3)).numpy()
    return labs, cmin, de


def curves(P, split, thumbs):
    st = C.Store(split, np.load(f'{C.DATA}/codebook_256.npy'))
    out = []
    for r in st.d:
        if r['id'] not in P['pred']:
            continue
        cols = P['pred'][r['id']].astype(np.float32)
        locked = P['locks'][r['id']]
        tp = [(i, j) for i, j in AC.text_pairs(r)]
        if not tp:
            continue
        audit = AC.prepare_audit(r, cols, thumbs)
        nT = len(tp)
        fail = np.zeros((len(LAMS), nT), bool)
        cost = np.zeros((len(LAMS), nT))
        edited = np.zeros((len(LAMS), nT), bool)
        proxy0 = np.zeros(nT, bool)
        none = AC.audit_fail(r, cols, audit, a.metric)
        het = np.array([AC.heterogeneity(r, cols, i) for i, _ in tp])
        last_choice = {}
        for t, ((i, j), au) in enumerate(zip(tp, audit)):
            labs, cmin, de = candidates(r, cols, i, j, a.metric)
            thr = AC.threshold(r, i, a.metric)
            proxy0[t] = cmin[0] < thr
            for k, lam in enumerate(LAMS):
                target = thr * (1 + lam)
                if cmin[0] >= target or locked[i]:
                    choice = 0
                else:
                    feas = cmin >= target
                    feas[0] = False
                    choice = int(np.argmin(np.where(feas, de, np.inf))) if feas.any() else int(np.argmax(cmin))
                c2 = cols.copy()
                c2[i] = labs[choice]
                if k == 0:
                    last_choice[t] = labs[choice]
                fail[k, t] = AC.audit_fail(r, c2, [au], a.metric)[0]
                cost[k, t] = de[choice]
                edited[k, t] = choice != 0
        plate_fail = np.zeros((len(OPAC), nT), bool)
        score = np.zeros(nT)
        scrim_cost = np.zeros(nT)
        for t, ((i, j), au) in enumerate(zip(tp, audit)):
            c1 = AC.to_srgb(last_choice[t])
            score[t] = AC.proxy_min_contrast(r, cols, c1, j, a.metric, i)[0] / AC.threshold(r, i, a.metric)
            # scrim: a black or white plate behind the text box, and the hue-preserving text candidate closest
            # to the generated colour whose contrast with the plate is at least 1.5 times the threshold
            labs_i, _, de_i = candidates(r, cols, i, j, a.metric)
            rgbs_i = AC.to_srgb(labs_i)
            thr_i = AC.threshold(r, i, a.metric)
            best = None
            for p in (WHITE, BLACK):
                cp = np.min(np.stack([AC.contrast(rgbs_i, np.broadcast_to(p, rgbs_i.shape), M, a.metric) for _, _, M in AC.COND_PROXY]), 0)
                ok = cp >= 1.5 * thr_i
                k_ = int(np.argmin(np.where(ok, de_i, np.inf))) if ok.any() else int(np.argmax(cp))
                cand = (0 if ok.any() else 1, de_i[k_] if ok.any() else -cp[k_], p, rgbs_i[k_], de_i[k_])
                if best is None or cand[:2] < best[:2]:
                    best = cand
            scrim_cost[t] = best[4]
            for k, o in enumerate(OPAC):
                plate_fail[k, t] = AC.audit_fail_plate(r, i, best[3], au['px'], best[2], o, a.metric)
        pl = np.zeros(NPLATE)
        sp = np.zeros(NPLATE)
        pl[0] = fail[0].mean()
        km = int(np.argmin(np.abs(OPAC - OPAC_MAIN)))
        for g, tau in enumerate(TAUS):
            scr = score < tau
            pl[1 + g] = np.where(scr, plate_fail[km], fail[0]).mean()
            sp[1 + g] = scr.mean()
        lp = np.zeros(NPATH)
        dp = np.zeros(NPATH)
        lp[0] = fail[0].mean()
        for g, tau in enumerate(TAUS):
            deferred = score < tau
            lp[1 + g] = (fail[0] & ~deferred).mean()
            dp[1 + g] = deferred.mean()
        out.append({'id': r['id'], 'group': r['group'], 'fail': fail, 'cost': cost, 'edited': edited, 'het': het, 'score': score,
                    'proxy0': proxy0, 'none': none, 'loss': lp, 'defer': dp, 'n_text': nT,
                    'loss_margin': fail.mean(1), 'scrim_cost': scrim_cost, 'loss_plate': pl, 'scrim': sp, 'plate_fail': plate_fail})
    return out


th = AC.Thumbs('prep', ('validation', 'test'))
Pc, Pt = pickle.load(open(a.cal, 'rb')), pickle.load(open(a.test, 'rb'))
cal = curves(Pc, 'validation', th)
test = curves(Pt, 'test', th)
pickle.dump({'lams': LAMS, 'taus': TAUS, 'opac': OPAC, 'opac_main': OPAC_MAIN, 'cal': cal, 'test': test, 'metric': a.metric}, open(a.out, 'wb'))
Lc = np.stack([d['loss'] for d in cal])
Lt = np.stack([d['loss'] for d in test])
PATH = np.arange(NPATH)
Dt = np.stack([d['defer'] for d in test])
for alpha in (0.05, 0.10, 0.15, 0.20):
    k, _ = AC.crc_lambda(Lc, PATH, alpha)
    k = int(k)
    desc = 'nominal repair' if k == 0 else f'nominal + defer s<{TAUS[k - 1]:.2f}'
    print(f'alpha {alpha:.2f} path {k} ({desc}) test risk {Lt[:, k].mean():.4f} defer {Dt[:, k].mean():.3f}')
print('no repair test risk', np.mean([d['none'].mean() for d in test]), 'nominal repair', Lt[:, 0].mean())

Lcp = np.stack([d['loss_plate'] for d in cal])
Ltp = np.stack([d['loss_plate'] for d in test])
for alpha in (0.05, 0.10, 0.15, 0.20):
    k, _ = AC.crc_lambda(Lcp, np.arange(NPLATE), alpha)
    k = int(k)
    desc = 'nominal repair' if k == 0 else f'nominal + scrim s<{TAUS[k - 1]:.2f}'
    St = np.stack([d['scrim'] for d in test])
    print(f'[scrim] alpha {alpha:.2f} path {k} ({desc}) test risk {Ltp[:, k].mean():.4f} scrimmed {St[:, k].mean():.3f}')
