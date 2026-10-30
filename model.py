"""Element-graph Transformer used by ChromaCert and its ablations.

The network reads a design as a set of layers. Each layer contributes its
geometry, role, fixed-colour statistics, frozen vision-language embedding and
(when observed or locked) its colour token. Pairwise structure enters through a
learned per-head attention bias indexed by the relation between two layers
(on-top-of, underneath, adjacent, same role) plus a slope on their overlap
fraction.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NREL = 6
N_TYPES = 5
N_FORMATS = 67
N_CATS = 23


class BiasedBlock(nn.Module):
    def __init__(self, d, h, p=0.1):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(p), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, bias):
        B, N, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.ln2(x)))


class ElementGraphNet(nn.Module):
    def __init__(self, K, d=192, layers=4, heads=6, feat_dim=24, sem_dim=512, use_rel=True,
                 use_sem=True, out='tokens', p=0.1):
        super().__init__()
        self.K, self.heads, self.use_rel, self.use_sem, self.out = K, heads, use_rel, use_sem, out
        self.feat = nn.Linear(feat_dim, d)
        self.etype = nn.Embedding(N_TYPES, d)
        self.tok = nn.Embedding(K + 2, d)               # K colours, MASK = K, NA = K + 1
        self.cont = nn.Linear(3, d)                     # continuous colour input (Gaussian baseline)
        self.fixed = nn.Linear(10, d)
        self.sem = nn.Linear(sem_dim, d)
        self.canvas = nn.Parameter(torch.zeros(d))
        self.fmt = nn.Embedding(N_FORMATS, d)
        self.cat = nn.Embedding(N_CATS, d)
        self.title = nn.Linear(sem_dim, d)
        self.time = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
        self.rel_bias = nn.Embedding(NREL + 1, heads)   # last id: canvas token links
        self.ovl_slope = nn.Parameter(torch.zeros(heads))
        self.blocks = nn.ModuleList([BiasedBlock(d, heads, p) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, K if out == 'tokens' else 3)
        nn.init.zeros_(self.rel_bias.weight)

    @staticmethod
    def t_embed(t, dim=64):
        f = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2))
        a = t[:, None] * 1000 * f[None]
        return torch.cat([a.sin(), a.cos()], -1)

    def forward(self, b, tokens=None, cont=None, t=None):
        x = self.feat(b['feat']) + self.etype(b["type"]) + self.fixed(torch.cat([b['fixed'], b['has_fixed'][..., None].float()], -1))
        if tokens is not None:
            x = x + self.tok(tokens)
        if cont is not None:
            x = x + self.cont(cont)
        if self.use_sem:
            x = x + self.sem(b['sem'])
        g = self.canvas + self.fmt(b['format']) + self.cat(b['category'])
        if self.use_sem:
            g = g + self.title(b['title_emb'])
        if t is not None:
            g = g + self.time(self.t_embed(t))
        x = torch.cat([g[:, None], x], 1)
        B, N1, _ = x.shape
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), b['valid']], 1)
        rel = torch.full((B, N1, N1), NREL, dtype=torch.long, device=x.device)
        rel[:, 1:, 1:] = b['rel'] if self.use_rel else (b['rel'] == 1).long()
        bias = self.rel_bias(rel).permute(0, 3, 1, 2)
        if self.use_rel:
            ovl = torch.zeros(B, N1, N1, device=x.device)
            ovl[:, 1:, 1:] = b['ovl']
            bias = bias + self.ovl_slope[None, :, None, None] * ovl[:, None]
        bias = bias.masked_fill(~valid[:, None, None, :], float('-inf'))
        for blk in self.blocks:
            x = blk(x, bias)
        return self.head(self.ln(x[:, 1:]))
