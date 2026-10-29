import sys; sys.path.insert(0, 'dqi')
import numpy as np
from divstep_model import *

def dg(p): return max([i for i, v in enumerate(p) if v] or [-1])

stats = {}
rng = np.random.default_rng(7)
for trial in range(300):
    w = int(rng.choice([35, 34, 33, 31, 28, 20, 5]))
    pos = rng.choice(m, w, replace=False)
    S = syndromes({int(p): int(rng.integers(1, 4096)) for p in pos})
    f = [1] + [0] * (n - 1); g = S[::-1]; delta = 1; bf, bg = [0], [1]
    for step in range(2 * ell):
        sw = delta > 0 and g[0] != 0
        if sw:
            f, g, bf, bg, delta = g, f, bg, bf, -delta
        c = mul(g[0], inv(f[0]))
        key = ('sw' if sw else 'ns', c != 0)
        d_bf, d_bg = dg(bf), dg(bg)
        rel = 'bg<bf' if d_bg < d_bf else ('bg=bf' if d_bg == d_bf else 'bg>bf')
        stats.setdefault(key, {}).setdefault((rel, d_bf - (step + 1 - delta) // 1 if False else d_bf - d_bg - delta), 0)
        stats[key][(rel, d_bf - d_bg - delta)] += 1
        L = max(len(bg), len(bf))
        bg = [(bg[i] if i < len(bg) else 0) ^ mul(c, bf[i] if i < len(bf) else 0) for i in range(L)]
        g = [g[i] ^ mul(c, f[i]) for i in range(len(g))][1:]
        f = f[:len(g)]
        bf = [0] + bf
        delta += 1
for k, v in stats.items():
    print(k, dict(sorted(v.items(), key=lambda t: -t[1])[:6]))
