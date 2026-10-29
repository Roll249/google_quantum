import sys; sys.path.insert(0, 'dqi'); sys.path.insert(0, 'data/verifiers/dqi')
import numpy as np
import gfalg as G
from divstep_model import mul, inv, poly_eval, n
import static_model as SM
import verify_submission_dqi as V
P = [70] + [70 - (s - 1) // 2 for s in range(1, 70)]
rng = np.random.default_rng(int(sys.argv[1]))
w_probs = V.optimal_dqi_weights(4095, 35, 12, 2016) ** 2
good = bad = ng = 0
for t in range(int(sys.argv[2])):
    w = int(rng.choice(36, p=w_probs)) if t % 3 else int(rng.integers(18, 36))
    pos = rng.choice(4095, w, replace=False)
    err = {int(p): int(rng.integers(1, 4096)) for p in pos}
    S = [0] * n
    for j, v in err.items():
        for i in range(n):
            S[i] ^= mul(v, G.gpow(2, ((i + 1) * j) % 4095))
    F, Gr, ok = SM.forward(S, P)
    if not ok:
        ng += 1; continue
    # swaps count = w for generic shots; emulate counter
    F2, G2 = SM.cleanup(F, Gr, w)
    if SM.chien(F2, G2) == err:
        good += 1
    else:
        bad += 1
        print('bad w', w)
print('good', good, 'bad', bad, 'non-generic', ng)
