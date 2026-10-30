"""In-context LLM colour completion following the ColorGPT protocol
(structured JSON layers, hex colours, one retrieved in-context example),
run with an open instruction-tuned model on CPU.

python llm_baseline.py --n 300 --model Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
import json
import os
import pickle
import re
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import baselines as BL
import chroma as C
import colorlib as cl

p = argparse.ArgumentParser()
p.add_argument('--n', type=int, default=300)
p.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
p.add_argument('--threads', type=int, default=2)
p.add_argument('--out', default='preds/llm_P_s0.pkl')
a = p.parse_args()
torch.set_num_threads(a.threads)
TYPES = ['vector shape', 'text', 'photo', 'background', 'masked shape']


def to_hex(oklab):
    rgb = cl.oklab_to_srgb(torch.tensor(oklab, dtype=torch.float32)).numpy()
    return '#%02x%02x%02x' % tuple(int(round(v * 255)) for v in np.clip(rgb, 0, 1))


def describe(r, cols, hide):
    els = []
    for i in range(r['n']):
        x, y, w, h = r['feat'][i, :4]
        e = {'id': i, 'type': TYPES[r['type'][i]], 'x': round(float(x - w / 2), 2), 'y': round(float(y - h / 2), 2),
             'w': round(float(w), 2), 'h': round(float(h), 2)}
        if r['type'][i] == 1:
            e['text'] = r['texts'][i][:40]
        if r['rec'][i]:
            e['color'] = '?' if hide[i] else to_hex(cols[i])
        elif r['has_fixed'][i]:
            e['average_color'] = to_hex(r['fixed'][i, :3])
        els.append(e)
    return {'title': r['title'][:80], 'elements': els}


cb = np.load(f'{C.DATA}/codebook_256.npy')
emb = pickle.load(open(f'{C.DATA}/emb_clip.pkl', 'rb'))
tr = C.Store('train', cb, emb)
te = C.Store('test', cb, emb)
Xtr = np.stack([np.asarray(r['title_emb'], np.float32) for r in tr.d])
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
model.eval()
rng = np.random.default_rng(0)
subset = sorted(rng.choice(len(te.d), a.n, replace=False).tolist())
recs = [te.d[i] for i in subset]
locks = C.design_locks(recs, 0.5, 0)
out, fails, t0 = {}, 0, time.time()
sysmsg = ('You are a graphic designer. Given a design as JSON, replace every "?" colour with a hex colour '
          'so that the design is harmonious and legible. Answer only with JSON mapping element id to hex colour.')
for r, lk in zip(recs, locks):
    ex = tr.d[int(np.argmax(Xtr @ np.asarray(r['title_emb'], np.float32)))]
    exl = C.design_locks([ex], 0.5, 0)[0]
    ex_hide = ex['rec'] & ~exl
    ex_ans = {str(i): to_hex(ex['col'][i]) for i in np.nonzero(ex_hide)[0]}
    hide = r['rec'] & ~lk
    msgs = [{'role': 'system', 'content': sysmsg},
            {'role': 'user', 'content': json.dumps(describe(ex, ex['col'], ex_hide))},
            {'role': 'assistant', 'content': json.dumps(ex_ans)},
            {'role': 'user', 'content': json.dumps(describe(r, r['col'], hide))}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt', return_dict=True)
    ids = enc['input_ids']
    with torch.no_grad():
        gen = model.generate(ids, attention_mask=enc['attention_mask'], max_new_tokens=16 * int(hide.sum()) + 16, do_sample=False)
    txt = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
    cols = r['col'].copy()
    found = dict(re.findall(r'"?(\d+)"?\s*:\s*"?(#[0-9a-fA-F]{6})', txt))
    for i in np.nonzero(hide)[0]:
        hx = found.get(str(i))
        if hx is None:
            fails += 1
            hx = '#808080'
        rgb = torch.tensor(cl.hex_to_rgb(hx))
        cols[i] = cl.srgb_to_oklab(rgb).numpy()
    out[r['id']] = cols
    if len(out) % 5 == 0:
        print(len(out), f'{(time.time() - t0) / len(out):.1f}s/design', 'missing', fails, flush=True)
el = time.time() - t0
pickle.dump({'pred': out, 'locks': {r['id']: l for r, l in zip(recs, locks)}, 'seconds': el,
             'per_design_ms': 1000 * el / len(recs), 'missing': fails, 'model': a.model},
            open(a.out, 'wb'))
print('done', fails)
