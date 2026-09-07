import math, random, itertools
from fractions import Fraction

def wilson(k, n, z=1.959963985):
    if n == 0: return (float('nan'),)*2
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0.0, c-h), min(1.0, c+h))

def _ranks(vals):                       # average ranks for ties
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0]*len(vals); i = 0
    while i < len(order):
        j = i
        while j+1 < len(order) and vals[order[j+1]] == vals[order[i]]: j += 1
        avg = (i+j)/2 + 1
        for k in range(i, j+1): r[order[k]] = avg
        i = j+1
    return r

def wilcoxon_exact(diffs):
    """Two-sided exact Wilcoxon signed-rank. Zeros dropped (Wilcoxon's own rule)."""
    nz = [d for d in diffs if d != 0]
    m = len(nz)
    if m == 0: return dict(n_nonzero=0, W=None, p=1.0, exact=True)
    R = _ranks([abs(d) for d in nz])
    Wp = sum(r for d, r in zip(nz, R) if d > 0)
    total = sum(R)
    obs = abs(Wp - total/2)
    if m > 20: return dict(n_nonzero=m, W=Wp, p=None, exact=False)
    hits = 0
    for signs in itertools.product((0, 1), repeat=m):
        w = sum(r for s, r in zip(signs, R) if s)
        if abs(w - total/2) >= obs - 1e-12: hits += 1
    return dict(n_nonzero=m, W=Wp, p=hits/2**m, exact=True)

def sign_exact(diffs):
    pos = sum(1 for d in diffs if d > 0); neg = sum(1 for d in diffs if d < 0)
    m = pos+neg
    if m == 0: return dict(pos=0, neg=0, p=1.0)
    k = min(pos, neg)
    p = sum(math.comb(m, i) for i in range(k+1)) / 2**m * 2
    return dict(pos=pos, neg=neg, p=min(1.0, p))

def boot_ci(diffs, B=20000, seed=0, lo=2.5, hi=97.5):
    rng = random.Random(seed); n = len(diffs); out = []
    for _ in range(B):
        out.append(sum(diffs[rng.randrange(n)] for _ in range(n))/n)
    out.sort()
    return out[int(B*lo/100)], out[int(B*hi/100)]

def paired(a, b):
    """a, b: dict block->value. Returns full paired report for b - a."""
    ks = sorted(set(a) & set(b))
    d = [b[k]-a[k] for k in ks]
    mean = sum(d)/len(d)
    return dict(n=len(ks), blocks=ks, diffs=d, mean=mean,
                ci=boot_ci(d), wilcoxon=wilcoxon_exact(d), sign=sign_exact(d))
