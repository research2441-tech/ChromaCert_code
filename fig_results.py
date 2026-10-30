"""Result figures (grayscale) from res/summary.json and companion files."""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 7.5,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'mathtext.fontset': 'dejavuserif', 'legend.frameon': False})
S = json.load(open('res/summary.json'))
R = S['rows']
os.makedirs('figs', exist_ok=True)
NAMES = [('freq', 'Frequency prior'), ('knn', 'Retrieval'), ('img_resnet50', 'ResNet-50 palette'),
         ('img_clip', 'CLIP ViT palette'), ('mcr', 'One-shot MCM'), ('gauss', 'Gaussian diffusion'),
         ('mdm', 'Masked diffusion'), ('full', 'ChromaCert')]
HATCH = ['', '////', '....', 'xxxx', '++', '----', '||', '']


def get(name, key):
    v = R.get(name, {}).get(key)
    return (np.nan, None) if v is None else (v[0], v[1])


def fig_main():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), gridspec_kw={'width_ratios': [1.05, 1], 'wspace': 0.42})
    ax = axes[0]
    names = [(k, n) for k, n in NAMES if not np.isnan(get(k, 'P_de')[0])]
    y = np.arange(len(names))
    for i, (k, n) in enumerate(names):
        m, s = get(k, 'P_de')
        ax.barh(i, m, xerr=s if s else None, color='white' if k != 'full' else '0.25', edgecolor='black', lw=0.6,
                hatch=HATCH[i % len(HATCH)] if k != 'full' else '', error_kw={'elinewidth': 0.7, 'capsize': 2})
        ax.text(m + (s or 0) + 0.6, i, f'{m:.1f}', va='center', fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels([n for _, n in names])
    ax.invert_yaxis()
    ax.set_xlabel('Completion error $\\overline{\\Delta E}_{00}$ (lower is better)')
    ax.set_xlim(0, max(get(k, 'P_de')[0] for k, _ in names) * 1.22)
    ax.set_title('(a) Protocol P: brand-locked completion', fontsize=7.5, loc='left')
    ax = axes[1]
    markers = ['s', 'D', '^', 'v', 'p', 'h', 'o', '*']
    pts = []
    for i, (k, n) in enumerate(NAMES):
        x, _ = get(k, 'G_c2st')
        yv, _ = get(k, 'G_pix_wcag')
        if np.isnan(x) or np.isnan(yv):
            continue
        ax.scatter(x, yv, marker=markers[i], s=46 if k == 'full' else 26, facecolor='black' if k == 'full' else 'white',
                   edgecolor='black', lw=0.7, zorder=3, label=n)
        pts.append((x, yv))
    dy = get('designer', 'G_pix_wcag')[0]
    if not np.isnan(dy):
        ax.axhline(dy, color='0.4', lw=0.7, ls=(0, (4, 2)))
        ax.text(0.505, dy + 0.012, 'designer colourings', fontsize=6.3, color='0.25')
    ax.axvline(0.5, color='0.6', lw=0.6, ls=':')
    ax.set_xlabel('Two-sample critic ROC-AUC (0.5 = indistinguishable)')
    ax.set_ylabel('Pixel-level failure rate (WCAG, 16 observers)')
    ax.set_title('(b) Protocol G: realism versus legibility', fontsize=7.5, loc='left')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=6.2, handletextpad=0.4, labelspacing=0.55,
              borderaxespad=0.0)
    for a in axes:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    fig.savefig('figs/fig2_main.png', dpi=600, bbox_inches='tight', pad_inches=0.03)


def fig_observers(path='res/obs_full_G_s0.json'):
    if not os.path.exists(path):
        return
    O = json.load(open(path))
    names = O['observers']
    sev = [0.2, 0.4, 0.6, 0.8, 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2), sharey=True, gridspec_kw={'wspace': 0.12})
    series = [('designer', 'Designer', 'o', '0.55', (0, (4, 2))), ('generated', 'Generated', 's', 'black', '-'),
              ('repaired', 'Nominal repair', '^', 'black', (0, (1, 1.5)))]
    for ax, kind, title in zip(axes, ('protan', 'deutan', 'tritan'), ('Protanomaly', 'Deuteranomaly', 'Tritanomaly')):
        for key, lab, mk, col, ls in series:
            if key not in O:
                continue
            v = [O[key][names.index(f'{kind}-{s:.1f}')] for s in sev]
            ax.plot(sev, v, marker=mk, color=col, ls=ls, lw=0.9, ms=4, mfc='white', label=lab)
            ax.axhline(O[key][names.index('normal')], color=col, lw=0.5, ls=':')
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel('Simulated severity')
        ax.set_xticks(sev)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('Text-layer failure rate (WCAG)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=6.4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.savefig('figs/fig3_observers.png', dpi=600, bbox_inches='tight', pad_inches=0.03)


def fig_crc(key='crc_full_s0_wcag'):
    C = S.get(key)
    if C is None:
        return
    lam = [0.0] + C['taus']
    D, Sc = C['policies']['defer'], C['policies']['scrim']
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), gridspec_kw={'wspace': 0.45})
    ax = axes[0]
    ax.plot(lam, D['curve_cal'], color='black', lw=1.0, label='deferral, calibration')
    ax.plot(lam, D['curve_test'], color='black', lw=1.0, ls=(0, (4, 2)), label='deferral, test')
    ax.plot(lam, Sc['curve_test'], color='0.5', lw=1.0, ls=(0, (1, 1.2)), label='scrim, test')
    for a, st in zip(['0.05', '0.1', '0.2'], [':', '--', '-.']):
        ax.axhline(float(a), color='0.55', lw=0.6, ls=st)
        ax.text(1.97, float(a) + 0.004, f'$\\alpha$ = {float(a):.2f}', fontsize=5.6, va='bottom', ha='right', color='0.3')
    ax.set_xlim(0, 2.0)
    ax.set_xlabel('Escalation threshold $\\lambda$')
    ax.set_ylabel('Mean illegible fraction')
    ax.legend(loc='center right', fontsize=5.8, bbox_to_anchor=(1.0, 0.6), frameon=False)
    ax.set_title('(a) Risk along the path', fontsize=7.5, loc='left')
    ax = axes[1]
    al = sorted(D['alphas'], key=float)
    data = [D['alphas'][a]['resplit_risks'] for a in al]
    ax.boxplot(data, positions=range(len(al)), widths=0.5, patch_artist=True, showfliers=False,
               medianprops={'color': 'black', 'lw': 0.8}, boxprops={'facecolor': 'white', 'lw': 0.7},
               whiskerprops={'lw': 0.7}, capprops={'lw': 0.7})
    ax.plot(range(len(al)), [float(a) for a in al], '_', color='0.45', ms=16, mew=1.0, label='target $\\alpha$', zorder=3)
    ax.plot(range(len(al)), [D['alphas'][a]['risk'] for a in al], 'ko', ms=3.5, label='validation$\\to$test')
    ax.set_xticks(range(len(al)))
    ax.set_xticklabels([f'{float(a):.2f}' for a in al])
    ax.set_xlabel('Target level $\\alpha$')
    ax.set_ylabel('Realised test risk')
    ax.legend(loc='upper left', fontsize=6.0)
    ax.set_title('(b) 200 random re-splits', fontsize=7.5, loc='left')
    ax = axes[2]
    xs = [float(a) for a in al]
    ax.plot(xs, [D['alphas'][a]['esc'] for a in al], 'k-s', ms=3.5, mfc='white', lw=0.9, label='deferred')
    ax.plot(xs, [Sc['alphas'][a]['esc'] for a in al], 'k--^', ms=3.5, mfc='black', lw=0.9, label='scrimmed')
    ax.axhline(C['nominal_edit'], color='0.5', lw=0.8, ls=':')
    ax.text(0.2, C['nominal_edit'] + 0.012, 'recoloured by\nnominal repair', fontsize=5.6, ha='right', va='bottom', color='0.3', linespacing=1.1)
    ax.set_xlabel('Target level $\\alpha$')
    ax.set_ylabel('Fraction of text layers')
    ax.legend(loc='upper right', fontsize=6.0, bbox_to_anchor=(1.0, 0.85))
    ax.set_title('(c) Price of certification', fontsize=7.5, loc='left')
    for a in axes:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    fig.savefig('figs/fig4_crc.png', dpi=600, bbox_inches='tight', pad_inches=0.03)


def fig_sens():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3), gridspec_kw={'wspace': 0.95})
    g = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    keys = ['dpo', 'full', 'sens_gamma0.5', 'sens_gamma1.0', 'sens_gamma2.0', 'sens_gamma4.0']
    de = [get(k, 'P_de')[0] if k not in ('dpo', 'full') else R[k]['P_de'][0] for k in keys]
    pix = [get(k, 'G_pix_wcag')[0] if k not in ('dpo', 'full') else R[k]['G_pix_wcag'][0] for k in keys]
    ax = axes[0]
    ax.plot(range(len(g)), de, 'k-o', ms=3.5, mfc='white', lw=0.9, label='$\\overline{\\Delta E}_{00}$ (P)')
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(['0', '0.25', '0.5', '1', '2', '4'], fontsize=6.4)
    ax.set_xlabel('Guidance strength $\\gamma$')
    ax.set_ylabel('Completion error')
    ax2 = ax.twinx()
    ax2.plot(range(len(g)), pix, color='0.45', ls='--', marker='s', ms=3.5, mfc='0.45', lw=0.9, label='pixel failure (G)')
    ax2.set_ylabel('Pixel failure rate')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper center', fontsize=6.0, bbox_to_anchor=(0.5, -0.26), ncol=1, frameon=False)
    ax.set_title('(a) Guidance', fontsize=7.5, loc='left')
    ax = axes[1]
    steps = [1, 2, 4, 8, 16]
    keys = ['sens_steps1', 'sens_steps2', 'sens_steps4', 'full', 'sens_steps16']
    de = [get(k, 'P_de')[0] for k in keys]
    B = json.load(open('res/bench.json')) if os.path.exists('res/bench.json') else {}
    ms = [B.get(f'chromacert_S{x}_guided', np.nan) for x in steps]
    ax.plot(range(len(steps)), de, 'k-o', ms=3.5, mfc='white', lw=0.9, label='$\\overline{\\Delta E}_{00}$ (P)')
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel('Unmasking steps $S$')
    ax.set_ylabel('Completion error')
    ax2 = ax.twinx()
    ax2.plot(range(len(steps)), ms, color='0.45', ls='--', marker='s', ms=3.5, mfc='0.45', lw=0.9, label='latency (ms)')
    ax2.set_ylabel('Latency per design (ms)')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper center', fontsize=6.0, bbox_to_anchor=(0.5, -0.26), ncol=1, frameon=False)
    ax.set_title('(b) Sampling steps', fontsize=7.5, loc='left')
    ax = axes[2]
    lf = [0.25, 0.5, 0.75]
    full = [get('sens_lock0.25', 'P_de')[0], R['full']['P_de'][0], get('sens_lock0.75', 'P_de')[0]]
    mcr = [get('sens_lock0.25_mcr', 'P_de')[0], R['mcr']['P_de'][0], get('sens_lock0.75_mcr', 'P_de')[0]]
    ax.plot(lf, full, 'k-o', ms=3.5, mfc='black', lw=0.9, label='ChromaCert')
    ax.plot(lf, mcr, 'k--s', ms=3.5, mfc='white', lw=0.9, label='One-shot MCM')
    ax.set_xticks(lf)
    ax.set_xlabel('Fraction of locked layers')
    ax.set_ylabel('Completion error')
    ax.legend(loc='upper center', fontsize=6.0, bbox_to_anchor=(0.5, -0.26), ncol=1, frameon=False)
    ax.set_title('(c) Brand constraints', fontsize=7.5, loc='left')
    for a in axes:
        a.spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    fig.savefig('figs/fig5_sensitivity.png', dpi=600, bbox_inches='tight', pad_inches=0.03)


def fig_groups(key='crc_full_s0_wcag', group_de=None):
    C = S.get(key)
    if C is None or group_de is None:
        return
    C = C['policies']['defer']
    GN = ['SM', 'HC', 'MM', 'SMA', 'EO', 'BG']
    gs = sorted(int(g) for g in C['group_risk'])
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.0), gridspec_kw={'wspace': 0.3})
    ax = axes[0]
    ax.bar(range(len(gs)), [group_de[str(g)] for g in gs], color='white', edgecolor='black', lw=0.6, hatch='////')
    ax.set_xticks(range(len(gs)))
    ax.set_xticklabels([GN[g] for g in gs])
    ax.set_ylabel('Completion error $\\overline{\\Delta E}_{00}$')
    ax.set_xlabel('Crello design group')
    ax.set_title('(a) Fidelity by group', fontsize=7.5, loc='left')
    ax = axes[1]
    ax.bar(range(len(gs)), [C['group_risk'][str(g)][0] for g in gs], color='white', edgecolor='black', lw=0.6, hatch='....')
    ax.axhline(0.10, color='black', lw=0.7, ls='--')
    ax.text(1.01, 0.10, '$\\alpha=0.10$', fontsize=6.3, ha='left', va='center', transform=ax.get_yaxis_transform())
    ax.set_xticks(range(len(gs)))
    ax.set_xticklabels([GN[g] for g in gs])
    ax.set_ylabel('Realised illegible fraction')
    ax.set_xlabel('Crello design group')
    ax.set_title('(b) Certified risk by group', fontsize=7.5, loc='left')
    for a in axes:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    fig.savefig('figs/fig6_groups.png', dpi=600, bbox_inches='tight', pad_inches=0.03)


if __name__ == '__main__':
    import sys
    fig_main()
    fig_observers()
    fig_crc()
    fig_sens()
    gd = json.load(open('res/group_de.json')) if os.path.exists('res/group_de.json') else None
    fig_groups(group_de=gd)
