import sys; sys.path.insert(0, 'dqi')
sys.argv = [sys.argv[0], '0', '0', '73']
import numpy as np, gfalg as G
from divstep_model import mul, inv, poly_eval, n
import general_model as GM
rng = np.random.default_rng(9)
base = [70] + [70 - (t - 1) // 2 for t in range(1, 70)]
P = base[:]
P[0] = 70
cnt = {'gen': 0, 'ng': 0}
while min(cnt.values()) < 4:
    w = int(rng.integers(20, 36))
    pos = rng.choice(4095, w, replace=False)
    err = {int(p): int(rng.integers(1, 4096)) for p in pos}
    S = [0] * n
    for j, v in err.items():
        for i in range(n):
            S[i] ^= mul(v, G.gpow(2, ((i + 1) * j) % 4095))
    F, Gr, delta, ok, kinds = GM.forward(S, [max(p, 0) for p in P])
    kind = 'ng' if '2' in kinds else 'gen'
    if cnt[kind] >= 4: continue
    cnt[kind] += 1
    K = GM.K
    nzF = [i for i in range(GM.N) if F[i]]; nzG = [i for i in range(GM.N) if Gr[i]]
    bgdeg = max(j for j in range(36) if F[K - j])
    # find f region end: largest i < K - bgdeg - 1 with F[i] != 0
    f_top = max(i for i in nzF if i < K - bgdeg)
    gs = [i for i in nzG]
    print(kind, 'w', w, 'delta', delta, '(71-d)/2', (71 - delta) / 2, 'deg bg', bgdeg, 'f top', f_top, 'G nz', nzG[:2], '...', [i for i in nzG if i < 40][-3:], 'G nz high', nzG[-3:], 'ok', ok)
