"""ChromaCert core: data batching, masked discrete diffusion training,
harmony-guided sampling, masked-diffusion preference optimisation and the
reference baselines that share the same backbone.
"""
import math
import os
import pickle
import random
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

import colorlib as cl
import harmony as hm
from model import ElementGraphNet

DATA = os.environ.get('CC_DATA', 'data')
T_TEXT, T_IMG = 1, 2


# ---------------------------------------------------------------------------
# data

class Store:
    """Holds one split together with tokenised colours and frozen embeddings."""

    def __init__(self, split, codebook, emb=None, sem_dim=512):
        self.d = pickle.load(open(f'{DATA}/designs_{split}.pkl', 'rb'))
        self.cb = torch.tensor(codebook)
        self.K = len(codebook)
        cbt = self.cb
        for r in self.d:
            c = torch.tensor(r['col'])
            tok = torch.cdist(c, cbt).argmin(1).numpy()
            r['tok'] = np.where(r['rec'], tok, self.K + 1)
            sem = np.zeros((r['n'], sem_dim), np.float16)
            if emb is not None:
                for i in range(r['n']):
                    v = emb['img'].get((r['id'], i)) if r['type'][i] == T_IMG else emb['txt'].get((r['id'], i))
                    if v is not None:
                        sem[i] = v
                r['title_emb'] = emb['txt'].get(('title', r['id']), np.zeros(sem_dim, np.float16))
            else:
                r['title_emb'] = np.zeros(sem_dim, np.float16)
            r['sem'] = sem

    def __len__(self):
        return len(self.d)


def collate(recs, device='cpu'):
    B = len(recs)
    N = max(r['n'] for r in recs)

    def pad(key, shape, dtype, fill=0):
        out = np.full((B, N) + shape, fill, dtype)
        for b, r in enumerate(recs):
            out[b, :r['n']] = r[key]
        return torch.from_numpy(out)

    rel = np.zeros((B, N, N), np.int64)
    ovl = np.zeros((B, N, N), np.float32)
    for b, r in enumerate(recs):
        rel[b, :r['n'], :r['n']] = r['rel']
        ovl[b, :r['n'], :r['n']] = r['ovl']
    batch = {
        'feat': pad('feat', (24,), np.float32), 'type': pad('type', (), np.int64),
        'fixed': pad('fixed', (9,), np.float32), 'has_fixed': pad('has_fixed', (), bool),
        'rec': pad('rec', (), bool), 'tok': pad('tok', (), np.int64, -1), 'col': pad('col', (3,), np.float32),
        'sem': pad('sem', (512,), np.float32), 'rel': torch.from_numpy(rel), 'ovl': torch.from_numpy(ovl),
        'format': torch.tensor([r['format'] for r in recs]), 'category': torch.tensor([r['category'] for r in recs]),
        'title_emb': torch.from_numpy(np.stack([r['title_emb'] for r in recs]).astype(np.float32)),
    }
    batch['valid'] = torch.zeros(B, N, dtype=torch.bool)
    for b, r in enumerate(recs):
        batch['valid'][b, :r['n']] = True
    return {k: v.to(device) for k, v in batch.items()}


def lock_mask(rec, frac, gen):
    """Designer-locked colours: a random subset of recolourable layers (brand constraints)."""
    lock = torch.zeros_like(rec)
    for b in range(rec.shape[0]):
        idx = torch.nonzero(rec[b]).flatten()
        k = int(round(frac * len(idx)))
        k = min(k, len(idx) - 1)
        if k > 0:
            perm = idx[torch.randperm(len(idx), generator=gen)[:k]]
            lock[b, perm] = True
    return lock



def pad_lock(recs, locks):
    N = max(r['n'] for r in recs)
    out = torch.zeros(len(recs), N, dtype=torch.bool)
    for b, (r, l) in enumerate(zip(recs, locks)):
        out[b, :r['n']] = torch.as_tensor(l, dtype=torch.bool)
    return out


def design_locks(recs, frac, seed):
    """Deterministic per-design lock patterns (a fraction of recolourable layers)."""
    out = []
    for r in recs:
        rng = np.random.default_rng(zlib.crc32(f"{r['id']}|{seed}|{frac:.3f}".encode()))
        idx = np.nonzero(r['rec'])[0]
        k = min(int(round(frac * len(idx))), len(idx) - 1)
        l = np.zeros(r['n'], bool)
        if k > 0:
            l[rng.choice(idx, k, replace=False)] = True
        out.append(l)
    return out


# ---------------------------------------------------------------------------
# training objectives

def masked_diffusion_loss(net, b, K, gen, lock_frac=None, weighting='elbo', reduce=True, t=None, m=None):
    """Absorbing-state (masked) diffusion ELBO with linear schedule alpha_t = 1 - t."""
    rec = b['rec']
    if lock_frac is None:
        lock_frac = torch.rand((), generator=gen).item() * 0.6
    lock = lock_mask(rec, lock_frac, gen) if m is None else m[1]
    free = rec & ~lock
    B = rec.shape[0]
    if t is None:
        t = torch.rand(B, generator=gen).clamp(0.02, 1.0)
    if m is None:
        u = torch.rand(rec.shape, generator=gen)
        mk = free & (u < t[:, None])
        # guarantee at least one masked free layer per design
        none = mk.sum(1) == 0
        if none.any():
            for bb in torch.nonzero(none).flatten():
                cand = torch.nonzero(free[bb]).flatten()
                if len(cand):
                    mk[bb, cand[torch.randint(len(cand), (1,), generator=gen)]] = True
    else:
        mk = m[0]
    tokens = b['tok'].clone().clamp_min(0)
    tokens = torch.where(mk, torch.full_like(tokens, K), tokens)
    logits = net(b, tokens=tokens)
    tgt = torch.where(rec, b['tok'], torch.zeros_like(b['tok']))
    nll = F.cross_entropy(logits.reshape(-1, K), tgt.reshape(-1), reduction='none').view(rec.shape)
    w = (1.0 / t)[:, None] if weighting == 'elbo' else torch.ones_like(t)[:, None]
    per = (nll * mk * w).sum(1) / free.sum(1).clamp_min(1)
    if reduce:
        return per.mean()
    return per, (mk, lock), t


def gaussian_diffusion_loss(net, b, gen, T=1000):
    """Continuous DDPM baseline in OKLab (epsilon prediction, cosine schedule)."""
    rec = b['rec']
    lock = lock_mask(rec, torch.rand((), generator=gen).item() * 0.6, gen)
    free = rec & ~lock
    B = rec.shape[0]
    t = torch.rand(B, generator=gen).clamp(1e-3, 1.0)
    abar = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    x0 = (b['col'] - COL_MU) / COL_SD
    eps = torch.randn(x0.shape, generator=gen)
    xt = abar.sqrt()[:, None, None] * x0 + (1 - abar).sqrt()[:, None, None] * eps
    xt = torch.where(free[..., None], xt, torch.where(lock[..., None], x0, torch.zeros_like(x0)))
    pred = net(b, cont=xt, tokens=torch.where(lock, b['tok'].clamp_min(0), torch.where(free, torch.full_like(b['tok'], net.K), b['tok'].clamp_min(0))), t=t)
    return (((pred - eps) ** 2).sum(-1) * free).sum() / free.sum().clamp_min(1)


COL_MU = torch.tensor([0.65, 0.0, 0.0])
COL_SD = torch.tensor([0.25, 0.08, 0.08])


def train(net, store, epochs, objective='mdm', bs=64, lr=3e-4, seed=0, log=None, ref=None, dpo=None):
    gen = torch.Generator().manual_seed(seed)
    random.seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(store) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=max(0.05, 3.0 / steps))
    idx = list(range(len(store)))
    hist = []
    for ep in range(epochs):
        random.shuffle(idx)
        net.train()
        tot, cnt, t0 = 0.0, 0, time.time()
        for s in range(0, len(idx), bs):
            b = collate([store.d[i] for i in idx[s:s + bs]])
            if objective == 'mdm':
                loss = masked_diffusion_loss(net, b, store.K, gen)
            elif objective == 'mlm':   # one-shot masked colour model (MCR-style baseline)
                loss = masked_diffusion_loss(net, b, store.K, gen, weighting='uniform')
            elif objective == 'gauss':
                loss = gaussian_diffusion_loss(net, b, gen)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b['feat'])
            cnt += len(b['feat'])
        hist.append(tot / cnt)
        if log:
            log(f'epoch {ep} loss {tot / cnt:.4f} {time.time() - t0:.0f}s')
    return hist


def dpo_finetune(net, ref, store, pairs, epochs=2, bs=32, lr=5e-5, beta=0.1, sft=1.0, seed=0, log=None):
    """Masked-diffusion preference optimisation.

    Each pair holds a design index, the lock pattern and a dispreferred token
    assignment of the free layers; the designer's own assignment is preferred.
    The implicit reward is the difference between policy and reference ELBO
    surrogates evaluated with a shared noise level and mask.
    """
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    ref.eval()
    order = list(range(len(pairs)))
    rng = random.Random(seed)
    hist = []
    for ep in range(epochs):
        rng.shuffle(order)
        net.train()
        tot, acc, cnt = 0.0, 0.0, 0
        for s in range(0, len(order), bs):
            ps = [pairs[i] for i in order[s:s + bs]]
            recs = [store.d[p['idx']] for p in ps]
            bw = collate(recs)
            bl = {k: v.clone() for k, v in bw.items()}
            lock = torch.zeros_like(bw['rec'])
            for j, p in enumerate(ps):
                n = recs[j]['n']
                lock[j, :n] = torch.from_numpy(p['lock'])
                bl['tok'][j, :n] = torch.from_numpy(p['lose'])
            free = bw['rec'] & ~lock
            t = torch.rand(len(ps), generator=gen).clamp(0.05, 1.0)
            mk = free & (torch.rand(free.shape, generator=gen) < t[:, None])
            mk = mk | (free & (mk.sum(1, keepdim=True) == 0))
            lw, _, _ = masked_diffusion_loss(net, bw, store.K, gen, reduce=False, t=t, m=(mk, lock))
            ll, _, _ = masked_diffusion_loss(net, bl, store.K, gen, reduce=False, t=t, m=(mk, lock))
            with torch.no_grad():
                rw, _, _ = masked_diffusion_loss(ref, bw, store.K, gen, reduce=False, t=t, m=(mk, lock))
                rl, _, _ = masked_diffusion_loss(ref, bl, store.K, gen, reduce=False, t=t, m=(mk, lock))
            margin = -beta * ((lw - rw) - (ll - rl))
            loss = -F.logsigmoid(margin).mean() + sft * lw.mean() * 0.1
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(ps)
            acc += (margin > 0).float().sum().item()
            cnt += len(ps)
        hist.append((tot / cnt, acc / cnt))
        if log:
            log(f'dpo epoch {ep} loss {tot / cnt:.4f} pref-acc {acc / cnt:.3f}')
    return hist


# ---------------------------------------------------------------------------
# guidance

class Guide:
    """Per-candidate harmony energy for one design given the current colour context."""

    def __init__(self, codebook, prior, lam_h=1.0, lam_s=1.0, lam_l=0.5):
        self.cb = torch.tensor(codebook)
        self.cb_lab = cl.srgb_to_cielab(cl.oklab_to_srgb(self.cb))
        self.cb_lch = cl.oklab_to_oklch(self.cb).numpy()
        self.cb_w = np.clip((self.cb_lch[:, 1] - hm.CHROMA_MIN) / 0.1, 0, 1)
        self.prior, self.lam_h, self.lam_s, self.lam_l = prior, lam_h, lam_s, lam_l

    def energies_all(self, r, cur, decided, idx):
        """Energies over all K candidates for every layer in idx (array of layer ids).

        cur: n x 3 OKLab colours (decided, expected, or fixed means); decided: bool n.
        The hue template is fitted once on the known context; separation and
        lightness terms use every known layer that overlaps the candidate layer."""
        K = len(self.cb)
        E = np.zeros((len(idx), K))
        n = r['n']
        known = decided | (~r['rec'] & r['has_fixed'])
        area = r['feat'][:, 2] * r['feat'][:, 3]
        if self.lam_h > 0 and known.sum() >= 1:
            h, w = hm.hue_weights(cur[known], area[known])
            tmpl, rot, _ = hm.fit_template(h, w)
            d = hm.sector_distance(self.cb_lch[:, 2], tmpl, rot) / (np.pi / 2) * 4
            E += self.lam_h * (self.cb_w * d)[None] * np.sqrt(np.maximum(area[idx], 1e-6))[:, None]
        pairs = []
        for a, i in enumerate(idx):
            for j in range(n):
                if j == i or not known[j]:
                    continue
                up, lo = (i, j) if j < i else (j, i)
                if r['ovl'][up, lo] >= 0.5:
                    pairs.append((a, i, j, up, lo))
        if pairs:
            labj = cl.srgb_to_cielab(cl.oklab_to_srgb(torch.tensor(cur[[p[2] for p in pairs]], dtype=torch.float32)))
            cbL = self.cb[:, 0].numpy()
            sep_rows = [k for k, p in enumerate(pairs) if r['type'][p[1]] == T_TEXT or r['type'][p[2]] == T_TEXT]
            if self.lam_s > 0 and sep_rows:
                A = self.cb_lab[None].expand(len(sep_rows), K, 3)
                Bv = labj[sep_rows][:, None].expand(len(sep_rows), K, 3)
                de = cl.ciede2000(A, Bv).numpy()
                for row, k in enumerate(sep_rows):
                    E[pairs[k][0]] += self.lam_s * hm.separation_energy(de[row])
            if self.lam_l > 0:
                for a, i, j, up, lo in pairs:
                    key = (int(r['type'][up]), int(r['type'][lo]))
                    dl = (cbL - cur[j, 0]) if up == i else (cur[j, 0] - cbL)
                    E[a] += self.lam_l * self.prior.energy(key, dl)
        return E


@torch.no_grad()
def sample(net, store, recs, lock, steps=8, guide=None, gamma=1.0, temp=1.0, seed=0, greedy=False):
    """Confidence-ordered iterative unmasking with optional harmony guidance.

    Returns a list of token arrays (length n per design)."""
    K = store.K
    gen = torch.Generator().manual_seed(seed)
    b = collate(recs)
    rec = b['rec']
    lock = pad_lock(recs, lock)
    free = rec & ~lock
    tokens = b['tok'].clone().clamp_min(0)
    tokens = torch.where(free, torch.full_like(tokens, K), tokens)
    masked = free.clone()
    total = free.sum(1)
    cb = store.cb
    for s in range(steps):
        logits = net(b, tokens=tokens) / temp
        logp = F.log_softmax(logits, -1)
        if guide is not None and gamma > 0:
            probs = logp.exp()
            exp_col = probs @ cb
            for bi, r in enumerate(recs):
                n = r['n']
                cur = np.where(r['rec'][:, None], r['col'], r['fixed'][:, :3]).astype(np.float32)
                dec = (~masked[bi, :n]).numpy() & r['rec']
                tk = tokens[bi, :n].numpy()
                cur = cur.copy()
                cur[dec] = cb[tk[dec]].numpy()
                mk = masked[bi, :n].numpy()
                cur[mk] = exp_col[bi, :n][torch.from_numpy(mk)].numpy()
                idx = np.nonzero(mk)[0]
                if len(idx):
                    E = guide.energies_all(r, cur, dec, idx)
                    logp[bi, torch.from_numpy(idx)] = logp[bi, torch.from_numpy(idx)] - gamma * torch.from_numpy(E).float()
            logp = F.log_softmax(logp, -1)
        if greedy:
            cand = logp.argmax(-1)
        else:
            g = -torch.log(-torch.log(torch.rand(logp.shape, generator=gen).clamp(1e-9, 1 - 1e-9)))
            cand = (logp + g).argmax(-1)
        conf = logp.gather(-1, cand[..., None]).squeeze(-1)
        conf = conf + (1 - (s + 1) / steps) * 1.0 * (-torch.log(-torch.log(torch.rand(conf.shape, generator=gen).clamp(1e-9, 1 - 1e-9))))
        conf = torch.where(masked, conf, torch.full_like(conf, -1e9))
        frac = 1 - math.cos(math.pi / 2 * (s + 1) / steps)
        target_unmasked = torch.ceil(total.float() * frac).long() if s < steps - 1 else total
        n_now = (free & ~masked).sum(1)
        k = (target_unmasked - n_now).clamp_min(0)
        order = conf.argsort(-1, descending=True)
        for bi in range(len(recs)):
            sel = order[bi, :k[bi]]
            sel = sel[masked[bi, sel]]
            tokens[bi, sel] = cand[bi, sel]
            masked[bi, sel] = False
    return [tokens[bi, :r['n']].numpy() for bi, r in enumerate(recs)]


@torch.no_grad()
def sample_gauss(net, store, recs, lock, steps=50, seed=0):
    """DDIM sampling for the continuous OKLab diffusion baseline; returns OKLab colours."""
    gen = torch.Generator().manual_seed(seed)
    b = collate(recs)
    rec = b['rec']
    lock = pad_lock(recs, lock)
    free = rec & ~lock
    x0_lock = (b['col'] - COL_MU) / COL_SD
    x = torch.randn(x0_lock.shape, generator=gen)
    ts = torch.linspace(1.0, 1e-3, steps + 1)
    toks = torch.where(lock, b['tok'].clamp_min(0), torch.where(free, torch.full_like(b['tok'], net.K), b['tok'].clamp_min(0)))
    for s in range(steps):
        t, tn = ts[s], ts[s + 1]
        ab = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        abn = torch.cos((tn + 0.008) / 1.008 * math.pi / 2) ** 2
        xin = torch.where(free[..., None], x, torch.where(lock[..., None], x0_lock, torch.zeros_like(x)))
        eps = net(b, cont=xin, tokens=toks, t=t.expand(len(recs)))
        x0 = ((xin - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-4, 4)
        x = abn.sqrt() * x0 + (1 - abn).sqrt() * eps
    out = x * COL_SD + COL_MU
    return [out[bi, :r['n']].numpy() for bi, r in enumerate(recs)]
