import sys; sys.path.insert(0, 'dqi'); sys.path.insert(0, 'data/verifiers/dqi')
import numpy as np
import verify_submission_dqi as V
from divstep_model import mul, inv, n, ell
import kickmix as km
spec = V.ProblemSpecDQI.benchmark_instance(); tables = V.FieldTables(km.GF2Field(12))
rng = np.random.default_rng(int(sys.argv[1]))
N = int(sys.argv[2])
syn, mb, ph = V.sample_shots(spec, tables, rng, N)
bad = 0; kinds = {}
for S in syn.tolist():
    f = [1] + [0] * (n - 1); g = S[::-1]; delta = 1; bf, bg = [0], [1]
    pat = []
    for step in range(2 * ell):
        sw = delta > 0 and g[0] != 0
        if sw:
            f, g, bf, bg, delta = g, f, bg, bf, -delta
        c = mul(g[0], inv(f[0]))
        pat.append('S' if sw else ('n' if c else '0'))
        L = max(len(bg), len(bf))
        bg = [(bg[i] if i < len(bg) else 0) ^ mul(c, bf[i] if i < len(bf) else 0) for i in range(L)]
        g = [g[i] ^ mul(c, f[i]) for i in range(len(g))][1:]
        f = f[:len(g)]
        bf = [0] + bf
        delta += 1
    p = ''.join(pat)
    k = p.rstrip('0')
    pairs = [p[i:i+2] for i in range(0, len(p), 2)]
    if any(q not in ("Sn", "S0", "00") for q in pairs):
        bad += 1
        kinds[k[:0] + str(len(k))] = kinds.get(str(len(k)), 0) + 1
        if bad <= 3:
            print(p)
print('non-generic', bad, 'of', N, bad / N)
