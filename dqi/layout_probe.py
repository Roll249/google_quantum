"""Find the static split P(n) and Omega location using the reference (unsplit) divstep model."""
import sys; sys.path.insert(0, 'dqi'); sys.path.insert(0, 'data/verifiers/dqi')
import numpy as np, kickmix as km
import gfalg as G
import verify_submission_dqi as V
from divstep_model import mul, inv, poly_eval, n

spec = V.ProblemSpecDQI.benchmark_instance(); tables = V.FieldTables(km.GF2Field(12))
rng = np.random.default_rng(int(sys.argv[1]))
Ns = int(sys.argv[2])
syn, mb, ph = V.sample_shots(spec, tables, rng, Ns)
# need error positions too: regenerate via our own sampler with the same distribution
w_probs = V.optimal_dqi_weights(4095, 35, 12, 2016) ** 2
lo = {}; hi = {}
omega_hits = {}
for t in range(Ns):
    w = int(rng.choice(36, p=w_probs))
    pos = rng.choice(4095, w, replace=False)
    err = {int(p): int(rng.integers(1, 4096)) for p in pos}
    S = [0] * n
    for j, v in err.items():
        for i in range(n):
            S[i] ^= mul(v, G.gpow(2, ((i + 1) * j) % 4095))
    f = [1]; g = S[::-1]; bf = [0]; bg = [1]
    ok = True
    for step in range(70):
        sw = (step % 2 == 0) and g[0] != 0
        if step % 2 == 1 and g[0] != 0 and False:
            pass
        if sw:
            f, g, bf, bg = g, f, bg, bf
        c = mul(g[0], inv(f[0]))
        dgf = lambda p: max([i for i, v in enumerate(p) if v] or [-1])
        if c:
            need_lo = max(dgf(f), dgf(g)) + 1       # P >= this
            need_hi = 71 - max(dgf(bf), dgf(bg))    # P <= K - max deg  (K = 71)
            lo[step] = max(lo.get(step, 0), need_lo)
            hi[step] = min(hi.get(step, 99), need_hi)
        L = max(len(f), len(g))
        f += [0] * (L - len(f)); g += [0] * (L - len(g))
        g = [g[i] ^ mul(c, f[i]) for i in range(L)][1:]
        Lb = max(len(bf), len(bg))
        bf += [0] * (Lb - len(bf)); bg += [0] * (Lb - len(bg))
        bg = [bg[i] ^ mul(c, bf[i]) for i in range(Lb)]
        bf = [0] + bf
    if not err:
        continue
    # bg(alpha^j) == 0 at error positions?
    X = {j: G.gpow(2, j) for j in err}
    if not all(poly_eval(bg, X[j]) == 0 for j in err):
        continue  # non-generic shot (wrong schedule)
    dbg = [(k * bg[k]) if False else (bg[k] if k % 2 == 1 else 0) for k in range(len(bg))]
    dbg = [bg[k] if k % 2 == 1 else 0 for k in range(1, len(bg))]  # formal derivative coefficients
    for name, P in (('f', f), ('g', g)):
        for a in range(-3, 4):
            vals = set()
            for j, v in err.items():
                num = poly_eval(P, X[j]); den = poly_eval(dbg, X[j])
                r = mul(mul(num, inv(den)), G.gpow(2, (a * j) % 4095)) if den else -1
                vals.add(mul(r, inv(v)) if r >= 0 else -1)
            if len(vals) == 1:
                omega_hits[(name, a)] = omega_hits.get((name, a), 0) + 1
print('split constraints per step (lo <= P <= hi):')
print(' '.join(f'{s}:{lo[s]}-{hi[s]}' for s in sorted(lo)))
print('omega candidates', omega_hits)
