"""lo/hi split constraints per step over many generic + degree-2 shots, for array size N."""
import sys, pickle; sys.path.insert(0, 'dqi'); sys.path.insert(0, 'data/verifiers/dqi')
import numpy as np
import gfalg as G
import verify_submission_dqi as V
from divstep_model import mul, inv
N = int(sys.argv[1]); K = N - 1
want = int(sys.argv[2])
rng = np.random.default_rng(int(sys.argv[3]))
wp = V.optimal_dqi_weights(4095, 35, 12, 2016) ** 2
exp, log = G.tables()
lo = {'g': [0] * 70, 'n': [0] * 70}; hi = {'g': [99] * 70, 'n': [99] * 70}
cnt = {'g': 0, 'n': 0}
dg = lambda p: max([i for i, v in enumerate(p) if v] or [-1])
while cnt['n'] < want:
    w = int(rng.choice(36, p=wp)); pos = rng.choice(4095, w, replace=False)
    vals = rng.integers(1, 4096, w)
    S = [0] * 70
    for j, v in zip(pos, vals):
        lv = log[int(v)]
        for i in range(70):
            S[i] ^= exp[(lv + (i + 1) * int(j)) % 4095]
    # quick generic test first (pattern), then full constraint pass only if needed
    f = [1]; g = S[::-1]; bf = [0]; bg = [1]; d = 1; pat = ''; rec = []
    for step in range(70):
        sw = d > 0 and g[0] != 0
        if sw: f, g, bf, bg, d = g, f, bg, bf, -d
        c = mul(g[0], inv(f[0])); pat += 'S' if sw else ('n' if c else '0')
        if c:
            rec.append((step, max(dg(f), dg(g)) + 1, K - max(dg(bf), dg(bg))))
        L = max(len(f), len(g)); f += [0] * (L - len(f)); g += [0] * (L - len(g))
        g = [g[i] ^ mul(c, f[i]) for i in range(L)][1:]
        Lb = max(len(bf), len(bg)); bf += [0] * (Lb - len(bf)); bg += [0] * (Lb - len(bg))
        bg = [bg[i] ^ mul(c, bf[i]) for i in range(Lb)]
        bf = [0] + bf
        d += 1
    pairs = [pat[i:i + 2] for i in range(0, 70, 2)]
    kind = 'g' if all(p in ('Sn', 'S0', '00') for p in pairs) else 'n'
    if kind == 'g' and cnt['g'] >= want:
        continue
    if kind == 'n' and '00S' in pat:   # degree >= 3: not supported
        continue
    cnt[kind] += 1
    for step, l, h in rec:
        lo[kind][step] = max(lo[kind][step], l); hi[kind][step] = min(hi[kind][step], h)
L = [max(lo['g'][s], lo['n'][s]) for s in range(70)]
H = [min(hi['g'][s], hi['n'][s]) for s in range(70)]
print('counts', cnt)
print('violations', [(s, L[s], H[s]) for s in range(70) if L[s] > H[s]])
print('P', [L[s] for s in range(70)])
pickle.dump((lo, hi), open(f'runs/split_{N}.pkl', 'wb'))
