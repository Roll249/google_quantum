import sys; sys.path.insert(0, 'dqi')
import numpy as np, gfalg as G
from divstep_model import mul, inv, poly_eval, n
rng = np.random.default_rng(9)
cnt = {'gen': 0, 'ng': 0}
dg = lambda p: max([i for i, v in enumerate(p) if v] or [-1])
tz = lambda p: min([i for i, v in enumerate(p) if v] or [-1])
while min(cnt.values()) < 4:
    w = int(rng.integers(20, 36))
    pos = rng.choice(4095, w, replace=False)
    err = {int(p): int(rng.integers(1, 4096)) for p in pos}
    S = [0] * n
    for j, v in err.items():
        for i in range(n):
            S[i] ^= mul(v, G.gpow(2, ((i + 1) * j) % 4095))
    f = [1]; g = S[::-1]; bf = [0]; bg = [1]; d = 1; pat = ''
    for step in range(70):
        sw = d > 0 and g[0] != 0
        if sw: f, g, bf, bg, d = g, f, bg, bf, -d
        c = mul(g[0], inv(f[0])); pat += 'S' if sw else ('n' if c else '0')
        L = max(len(f), len(g)); f += [0] * (L - len(f)); g += [0] * (L - len(g))
        g = [g[i] ^ mul(c, f[i]) for i in range(L)][1:]
        Lb = max(len(bf), len(bg)); bf += [0] * (Lb - len(bf)); bg += [0] * (Lb - len(bg))
        bg = [bg[i] ^ mul(c, bf[i]) for i in range(Lb)]
        bf = [0] + bf
        d += 1
    kind = 'ng' if any(pat[i:i + 2] not in ('Sn', 'S0', '00') for i in range(0, 70, 2)) else 'gen'
    if cnt[kind] >= 4: continue
    cnt[kind] += 1
    print(kind, 'w', w, 'deg f', dg(f), '(70-w)', 70 - w, 'deg bg', dg(bg), 'deg g', dg(g), 'deg bf', dg(bf), 'tz bf', tz(bf), 'h len', dg(bf) - tz(bf) + 1)
