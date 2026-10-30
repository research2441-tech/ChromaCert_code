"""Statistics helpers: seed aggregation, paired Wilcoxon tests with Benjamini-Hochberg
correction, matched-pairs rank-biserial effect size and paired bootstrap intervals."""
import numpy as np
from scipy.stats import wilcoxon, rankdata


def per_design(res, key):
    """Per-design scalar for a metric; list-valued accessibility entries become per-design means."""
    v = res[key]
    if isinstance(v, list):
        return np.array([np.mean(x) if len(x) else np.nan for x in v], float)
    return np.asarray(v, float)


def paired(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return a[m], b[m]


def rank_biserial(a, b):
    d = a - b
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def wilcoxon_p(a, b):
    a, b = paired(a, b)
    if np.allclose(a, b):
        return 1.0
    return float(wilcoxon(a, b, zero_method='wilcox', alternative='two-sided').pvalue)


def bootstrap_ci(a, b, n=2000, seed=0):
    a, b = paired(a, b)
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1], 1):
        k = n - rank + 1
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q
