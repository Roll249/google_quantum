"""Classical model of the general per-step static divstep decoder (supports quotients of degree <= 2)."""
import sys; sys.path.insert(0, 'dqi')
import numpy as np
import gfalg as G
from divstep_model import mul, inv, poly_eval, n
import static_model as SM

N = int(sys.argv[3]) if len(sys.argv) > 3 else 73
K = N - 1
SM.N, SM.K = N, K


def step_recover(F, Gr, delta_after):
    bg = lambda j: F[K - j]
    bf = lambda j: Gr[K - j]
    if bf(1):
        return mul(bg(0), inv(bf(1))), 'S'
    if delta_after >= 2:
        return 0, '0'
    if bf(2):
        iv = inv(bf(2))
        c0 = mul(bg(0), iv)
        return mul(bg(1) ^ mul(c0, bf(3)), iv), 'n1'
    if bf(3):
        iv = inv(bf(3))
        B = lambda j: bf(j + 3)
        c0 = mul(bg(0), iv)
        c1 = mul(bg(1) ^ mul(c0, B(1)), iv)
        return mul(bg(2) ^ mul(c0, B(2)) ^ mul(c1, B(1)), iv), 'n2'
    return None, '?'


def forward(S, P):
    F = [0] * N; Gr = [0] * N
    F[0] = 1; F[K] = 1
    for i in range(70):
        Gr[i] = S[69 - i]
    delta = 1
    ok = True
    kinds = ''
    for step in range(70):
        s = delta > 0 and Gr[0] != 0
        if s:
            F, Gr = Gr, F
            delta = 1 - delta
        else:
            delta += 1
        c = mul(Gr[0], inv(F[0]))
        p = P[step]
        for i in range(N):
            if i < p:
                Gr[i] ^= mul(c, F[i])
            else:
                F[i] ^= mul(c, Gr[i])
        if Gr[0] != 0:
            ok = False
        Gr = Gr[1:] + [0]
        rc, kind = step_recover(F, Gr, delta)
        kinds += kind[0] if kind != 'n2' else '2'
        if rc != c or ((Gr[K - 1] != 0) != s):
            ok = False
    return F, Gr, delta, ok, kinds


if __name__ == '__main__':
    sys.path.insert(0, 'data/verifiers/dqi')
    import verify_submission_dqi as V
    rng = np.random.default_rng(int(sys.argv[1]))
    w_probs = V.optimal_dqi_weights(4095, 35, 12, 2016) ** 2
    shots = []
    want_ng = int(sys.argv[2])
    tried = 0
    while sum(1 for s in shots if s[2]) < want_ng:
        tried += 1
        w = int(rng.choice(36, p=w_probs))
        pos = rng.choice(4095, w, replace=False)
        err = {int(p): int(rng.integers(1, 4096)) for p in pos}
        S = [0] * n
        for j, v in err.items():
            for i in range(n):
                S[i] ^= mul(v, G.gpow(2, ((i + 1) * j) % 4095))
        # true BY pattern
        f = [1]; g = S[::-1]; d = 1; pat = ''
        for step in range(70):
            sw = d > 0 and g[0] != 0
            if sw: f, g, d = g, f, -d
            c = mul(g[0], inv(f[0])); pat += 'S' if sw else ('n' if c else '0')
            L = max(len(f), len(g)); f += [0] * (L - len(f)); g += [0] * (L - len(g))
            g = [g[i] ^ mul(c, f[i]) for i in range(L)][1:]
            d += 1
        nong = any(pat[i:i + 2] not in ('Sn', 'S0', '00') for i in range(0, 70, 2))
        if nong or len(shots) < 40:
            shots.append((err, S, nong, pat))
    print('tried', tried, 'collected', len(shots))
    # split constraints over all collected shots (true BY dynamics, untruncated)
    lo = [0] * 70; hi = [99] * 70
    for err, S, nong, pat in shots:
        f = [1]; g = S[::-1]; bf = [0]; bg = [1]; d = 1
        dg = lambda p: max([i for i, v in enumerate(p) if v] or [-1])
        for step in range(70):
            if d > 0 and g[0] != 0:
                f, g, bf, bg, d = g, f, bg, bf, -d
            c = mul(g[0], inv(f[0]))
            if c:
                lo[step] = max(lo[step], max(dg(f), dg(g)) + 1)
                hi[step] = min(hi[step], K - max(dg(bf), dg(bg)))
            L = max(len(f), len(g)); f += [0] * (L - len(f)); g += [0] * (L - len(g))
            g = [g[i] ^ mul(c, f[i]) for i in range(L)][1:]
            Lb = max(len(bf), len(bg)); bf += [0] * (Lb - len(bf)); bg += [0] * (Lb - len(bg))
            bg = [bg[i] ^ mul(c, bf[i]) for i in range(Lb)]
            bf = [0] + bf
            d += 1
    viol = [s for s in range(70) if lo[s] > hi[s]]
    print('violations', viol)
    base = [70] + [70 - (t - 1) // 2 for t in range(1, 70)]
    P = [max(lo[s], min(hi[s], base[s])) for s in range(70)]
    print("P", P); print("base-ok", all(lo[s] <= base[s] <= hi[s] for s in range(70))); print("slack", [(s, lo[s], hi[s]) for s in range(70) if hi[s]-lo[s] < 1])
    good = bad = 0
    for err, S, nong, pat in shots:
        F, Gr, delta, ok, kinds = forward(S, P)
        wv = (71 - delta) // 2
        F2, G2 = SM.cleanup(F, Gr, wv) if True else (F, Gr)
        dec = SM.chien(F2, G2) if ok else None
        if ok and dec == err:
            good += 1
        else:
            bad += 1
            print('FAIL nong', nong, 'ok', ok, pat, kinds)
    print('good', good, 'bad', bad)
