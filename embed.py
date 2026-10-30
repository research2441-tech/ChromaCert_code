"""Frozen encoders: CLIP ViT-B/32 (image + text) for ChromaCert and
ResNet-50 / ViT-B/16 image features for the image-level baselines.

Usage: python embed.py clip|resnet50|vitb16
"""
import glob
import io
import pickle
import sys

import numpy as np
import torch
from PIL import Image

import os
torch.set_num_threads(int(os.environ.get("NT", "2")))
PREP, OUT = 'prep', 'data'


def crops():
    out = {}
    for f in sorted(glob.glob(f'{PREP}/*_crops.pkl')):
        out.update(pickle.load(open(f, 'rb')))
    return out


def designs():
    out = []
    for f in sorted(glob.glob(f'{PREP}/*_designs.pkl')):
        out.extend(pickle.load(open(f, 'rb')))
    return out


def encode_text_trunc(model, text):
    """CLIP text encoder on a batch truncated to its longest sequence (causal mask => same output)."""
    import open_clip.model as om
    L = text.shape[1]
    x = model.token_embedding(text)
    x = x + model.positional_embedding[:L]
    x = model.transformer(x, attn_mask=model.attn_mask[:L, :L])
    x = model.ln_final(x)
    x = om.text_global_pool(x, text, model.text_pool_type, eos_token_id=getattr(model, 'text_eos_id', None))
    if isinstance(model.text_projection, torch.nn.Linear):
        return model.text_projection(x)
    return x @ model.text_projection


@torch.no_grad()
def run_images(model, prep, items, bs=64):
    keys = list(items)
    feats = {}
    for s in range(0, len(keys), bs):
        kk = keys[s:s + bs]
        x = torch.stack([prep(Image.open(io.BytesIO(items[k])).convert('RGB')) for k in kk])
        y = model(x).float()
        y = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        for k, v in zip(kk, y.numpy().astype(np.float16)):
            feats[k] = v
        if s % (bs * 20) == 0:
            print(s, len(keys), flush=True)
    return feats


def main(which):
    items = crops()
    if which == 'clip':
        import open_clip
        model, _, prep = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
        tok = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
        model.eval()
        img = run_images(model.encode_image, prep, items)
        pickle.dump(img, open(f'{OUT}/emb_clip_img.pkl', 'wb'))
        texts = {}
        ds = designs()
        strings = []
        for d in ds:
            strings.append((('title', d['id']), (d['title'] + '. ' + ', '.join(d['keywords'][:20]))[:300]))
            for i, e in enumerate(d['elements']):
                if e['type'] == 1 and e['text'].strip():
                    strings.append(((d['id'], i), e['text'][:200]))
        uniq = sorted(set(s_ for _, s_ in strings))
        toks = tok(uniq)                                   # N x 77, zero padded after EOT
        lens = (toks != 0).sum(1)
        order = torch.argsort(lens)
        cache = {}
        with torch.no_grad():
            for s in range(0, len(order), 256):
                ii = order[s:s + 256]
                L = int(lens[ii].max())
                # causal text encoder: truncating padding after the EOT token leaves the pooled output unchanged
                y = encode_text_trunc(model, toks[ii][:, :L]).float()
                y = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                for k, v in zip(ii.tolist(), y.numpy().astype(np.float16)):
                    cache[uniq[k]] = v
                if s % 10240 == 0:
                    print('text', s, len(uniq), flush=True)
        for k, s_ in strings:
            texts[k] = cache[s_]
        pickle.dump({'img': img, 'txt': texts}, open(f'{OUT}/emb_clip.pkl', 'wb'))
    else:
        import timm
        name = {'resnet50': 'resnet50.a1_in1k', 'vitb16': 'vit_base_patch16_224.augreg2_in21k_ft_in1k'}[which]
        model = timm.create_model(name, pretrained=True, num_classes=0).eval()
        cfg = timm.data.resolve_data_config({}, model=model)
        prep = timm.data.create_transform(**cfg)
        part = os.environ.get('PART')
        if part:
            k, n = [int(x) for x in part.split('/')]
            keys = sorted(items)[k::n]
            items = {kk: items[kk] for kk in keys}
            img = run_images(model, prep, items, bs=32)
            pickle.dump({'img': img}, open(f'{OUT}/emb_{which}_part{k}.pkl', 'wb'))
            return
        img = run_images(model, prep, items, bs=32)
        pickle.dump({'img': img}, open(f'{OUT}/emb_{which}.pkl', 'wb'))


if __name__ == '__main__':
    main(sys.argv[1])
