"""Extract colour-relevant structure from one Crello v5.1 parquet shard.

For every template the script keeps canvas metadata, element geometry, the
designer-assigned colour of each recolourable element, a robust colour summary
of fixed elements (photographs and multi-colour vector art), small RGBA layer
thumbnails for pixel-level legibility checks (validation and test only) and
224-pixel JPEG crops of photographic elements for the frozen image encoder.

Usage: python prep_crello.py <shard.parquet> <split> <out_dir>
"""
import io
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

T_SVG, T_TEXT, T_IMG, T_BG, T_MASK = 0, 1, 2, 3, 4
CANVAS = 256  # longest canvas side used for pixel-level compositing
_rgba = re.compile(r'rgba?\(([^)]*)\)')


def parse_rgba(s):
    m = _rgba.match(s.strip())
    if not m:
        s = s.strip().lstrip('#')
        if len(s) >= 6:
            return [int(s[i:i + 2], 16) for i in (0, 2, 4)] + [1.0]
        return None
    p = [float(v) for v in m.group(1).split(',')]
    if len(p) == 3:
        p.append(1.0)
    return [int(round(p[0])), int(round(p[1])), int(round(p[2])), p[3]]


def decode(img_struct):
    if img_struct is None or img_struct.get('bytes') is None:
        return None
    try:
        return Image.open(io.BytesIO(img_struct['bytes'])).convert('RGBA')
    except Exception:
        return None


def summarize(im):
    """Alpha-weighted colour statistics of a rendered layer."""
    small = im.copy()
    small.thumbnail((96, 96))
    a = np.asarray(small, dtype=np.float32) / 255.0
    w = a[..., 3].reshape(-1)
    rgb = a[..., :3].reshape(-1, 3)
    keep = w > 0.5
    if keep.sum() < 4:
        return None
    rgb, w = rgb[keep], w[keep]
    mean = (rgb * w[:, None]).sum(0) / w.sum()
    q = np.quantile(rgb, [0.1, 0.5, 0.9], axis=0)
    cov = float(keep.mean())
    return {'mean': mean.tolist(), 'q10': q[0].tolist(), 'q50': q[1].tolist(),
            'q90': q[2].tolist(), 'coverage': cov}


def main(path, split, out_dir):
    pf = pq.ParquetFile(path)
    keep_thumbs = split in ('validation', 'test')
    designs, thumbs, crops = [], {}, {}
    cols = [c for c in pf.schema_arrow.names if c != 'preview']
    for rg in range(pf.metadata.num_row_groups):
        tab = pf.read_row_group(rg, columns=cols)
        for row in tab.to_pylist():
            cw, ch = float(row['canvas_width']), float(row['canvas_height'])
            scale = CANVAS / max(cw, ch)
            els, lay = [], []
            for i, t in enumerate(row['type']):
                x, y = row['left'][i], row['top'][i]
                w, h = row['width'][i], row['height'][i]
                e = {'type': int(t), 'box': [x / cw, y / ch, w / cw, h / ch],
                     'angle': float(row['angle'][i]), 'opacity': float(row['opacity'][i]),
                     'font_size': float(row['font_size'][i]) / ch, 'text': row['text'][i] or '',
                     'font': int(row['font'][i]), 'color': None, 'palette_len': 0,
                     'recolorable': False, 'summary': None}
                if t == T_TEXT:
                    cs = [parse_rgba(c) for c in (row['text_color'][i] or [])]
                    cs = [tuple(c[:3]) for c in cs if c is not None]
                    if cs:
                        e['color'] = list(Counter(cs).most_common(1)[0][0])
                        e['recolorable'] = True
                        e['n_text_colors'] = len(set(cs))
                    bold = row['font_bold'][i] or []
                    e['bold'] = float(np.mean(bold)) if bold else 0.0
                elif t in (T_BG, T_SVG, T_MASK):
                    pal = [parse_rgba(c) for c in (row['color'][i] or [])]
                    pal = [c for c in pal if c is not None]
                    e['palette_len'] = len(pal)
                    if len(pal) == 1:
                        e['color'] = pal[0][:3]
                        e['recolorable'] = True
                need_img = (not e['recolorable']) or keep_thumbs or t == T_IMG
                im = decode(row['image'][i]) if need_img else None
                if im is not None:
                    if not e['recolorable']:
                        e['summary'] = summarize(im)
                    if t == T_IMG:
                        c = im.convert('RGB')
                        c.thumbnail((224, 224))
                        buf = io.BytesIO()
                        c.save(buf, format='JPEG', quality=88)
                        crops[(row['id'], i)] = buf.getvalue()
                    if keep_thumbs:
                        tw = max(1, int(round(w * scale)))
                        th = max(1, int(round(h * scale)))
                        lay.append((i, np.asarray(im.resize((tw, th), Image.BILINEAR), dtype=np.uint8)))
                els.append(e)
            d = {'id': row['id'], 'split': split, 'group': int(row['group']), 'format': int(row['format']),
                 'category': int(row['category']), 'cw': cw, 'ch': ch, 'title': row['title'] or '',
                 'keywords': row['keywords'] or [], 'industries': row['industries'] or [],
                 'cluster': int(row['cluster_index']), 'elements': els}
            designs.append(d)
            if keep_thumbs:
                thumbs[row['id']] = lay
    base = os.path.join(out_dir, os.path.basename(path).replace('.parquet', ''))
    with open(base + '_designs.pkl', 'wb') as f:
        pickle.dump(designs, f)
    with open(base + '_crops.pkl', 'wb') as f:
        pickle.dump(crops, f)
    if keep_thumbs:
        with open(base + '_thumbs.pkl', 'wb') as f:
            pickle.dump(thumbs, f)
    print(path, len(designs), 'designs', len(crops), 'crops')


if __name__ == '__main__':
    main(*sys.argv[1:4])
