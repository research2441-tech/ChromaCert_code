"""Qualitative figure in grayscale: relative luminance of designer, generated and
certified colourings, plus the generated design as seen by a full-severity deuteranope.

python fig_qual.py preds/full_G_s0.pkl id1 id2 id3
"""
import pickle
import sys
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')
import access as AC
import chroma as C

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 7})


def render(r, cols, thumbs, M=np.eye(3)):
    layers = dict(thumbs.t.get(r['id'], []))
    scale = 256.0 / max(r['cw'], r['ch'])
    W, H = int(round(r['cw'] * scale)), int(round(r['ch'] * scale))
    canvas = np.ones((H, W, 3), np.float32)
    srgb = AC.to_srgb(cols)
    for j in range(r['n']):
        if j not in layers:
            continue
        im = layers[j].astype(np.float32) / 255.0
        x = r['feat'][j, 0] - r['feat'][j, 2] / 2
        y = r['feat'][j, 1] - r['feat'][j, 3] / 2
        ox, oy = int(round(x * W)), int(round(y * H))
        hh, ww = im.shape[:2]
        ax0, ay0, ax1, ay1 = max(0, ox), max(0, oy), min(W, ox + ww), min(H, oy + hh)
        if ax1 <= ax0 or ay1 <= ay0:
            continue
        patch = im[ay0 - oy:ay1 - oy, ax0 - ox:ax1 - ox]
        rgb, a = patch[..., :3], patch[..., 3:4] * r['opacity'][j]
        if r['rec'][j]:
            rgb = np.broadcast_to(srgb[j], rgb.shape)
        canvas[ay0:ay1, ax0:ax1] = rgb * a + canvas[ay0:ay1, ax0:ax1] * (1 - a)
    v = AC.view(canvas.reshape(-1, 3), M).reshape(canvas.shape)
    return AC.lum_wcag(AC.lin(v))


def main(pred_path, crc_path, ids, out):
    P = pickle.load(open(pred_path, 'rb'))
    st = C.Store('test', np.load(f'{C.DATA}/codebook_256.npy'))
    byid = {r['id']: r for r in st.d}
    th = AC.Thumbs('prep', ('test',))
    lam = float(crc_path)
    deut = [M for k, s, M in AC.COND_FULL if k == 'deutan' and s == 1.0][0]
    cols_titles = ['Designer', 'Generated', 'Certified repair', 'Generated, deuteranope']
    asp = [byid[i]['ch'] / byid[i]['cw'] for i in ids]
    pw = 7.16 / 4
    fig, axes = plt.subplots(len(ids), 4, figsize=(7.16, sum(pw * a for a in asp) + 0.35 * len(ids)),
                             gridspec_kw={'wspace': 0.06, 'hspace': 0.12, 'height_ratios': asp})
    for row, i in enumerate(ids):
        r = byid[i]
        gen = P['pred'][i]
        rep, _ = AC.repair(r, gen, lam, 'wcag')
        imgs = [render(r, r['col'], th), render(r, gen, th), render(r, rep, th), render(r, gen, th, deut)]
        for c, im in enumerate(imgs):
            ax = axes[row, c]
            ax.imshow(im, cmap='gray', vmin=0, vmax=1, interpolation='lanczos')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if row == 0:
                ax.set_title(cols_titles[c], fontsize=7.5, pad=4)
    fig.savefig(out, dpi=600, bbox_inches='tight', pad_inches=0.02)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:-1], sys.argv[-1])
