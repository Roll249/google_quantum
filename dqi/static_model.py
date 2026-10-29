"""Exact classical model of the static divstep decoder (layout F=(f,bg), G=(g,bf), N slots)."""
import sys; sys.path.insert(0, 'dqi')
import numpy as np
import gfalg as G
from divstep_model import mul, inv, poly_eval, n

N = 73
K = N - 1


def splits_from(lo):
    return lo


def forward(S, P):
    """Returns (F, G, ok) after 35 pairs; ok=False if a consistency check fails (non-generic shot)."""
    F = [0] * N; Gr = [0] * N
    F[0] = 1; F[K] = 1
    for i in range(70):
        Gr[i] = S[69 - i]
    ok = True
    for pair in range(35):
        cs = []
        s = Gr[0] != 0
        for half in range(2):
            step = 2 * pair + half
            if half == 0 and s:
                F, Gr = Gr, F
            c = mul(Gr[0], inv(F[0]))
            cs.append(c)
            p = P[step]
            for i in range(N):
                if i < p:
                    Gr[i] ^= mul(c, F[i])
                else:
                    F[i] ^= mul(c, Gr[i])
            if Gr[0] != 0:
                ok = False
            Gr = Gr[1:] + [0]
        # pair-end recovery: c' = bg_0 / bf_2 ; c = (bg_1 + c' bf_3) / bf_2 ; s = (bf_2 != 0)
        bg = lambda j: F[K - j]
        bf = lambda j: Gr[K - j]
        iv = inv(bf(2))
        c1 = mul(bg(0), iv)
        c2 = mul(bg(1) ^ mul(c1, bf(3)), iv)
        if (c1, c2) != tuple(cs) or (bf(2) != 0) != s:
            ok = False
    return F, Gr, ok


def decode(F, Gr):
    bg = [F[K - j] for j in range(36)]
    g = Gr[:36]
    errs = {}
    for j in range(4095):
        X = G.gpow(2, j)
        if poly_eval(bg, X) == 0:
            D = 0
            for k in range(1, 36, 2):
                D ^= mul(bg[k], G.gpow(X, k))
            errs[j] = mul(poly_eval(g, X), inv(D))
    return errs


if __name__ == '__main__':
    sys.path.insert(0, 'data/verifiers/dqi')
    import verify_submission_dqi as V
    rng = np.random.default_rng(int(sys.argv[1]))
    Ns = int(sys.argv[2])
    w_probs = V.optimal_dqi_weights(4095, 35, 12, 2016) ** 2
    shots = []
    for t in range(Ns):
        w = int(rng.choice(36, p=w_probs))
        pos = rng.choice(4095, w, replace=False)
        err = {int(p): int(rng.integers(1, 4096)) for p in pos}
        S = [0] * n
        for j, v in err.items():
            for i in range(n):
                S[i] ^= mul(v, G.gpow(2, ((i + 1) * j) % 4095))
        shots.append((err, S))
    # split: lo from probe at N=72 analysis -> recompute with K=72: P(step) = max remainder length
    lo = [0] * 70; hi = [99] * 70
    def generic(S):
        f = [1]; g = S[::-1]; delta = 1; pat = ''
        for step in range(70):
            sw = delta > 0 and g[0] != 0
            if sw: f, g, delta = g, f, -delta
            c = mul(g[0], inv(f[0])); pat += 'S' if sw else ('n' if c else '0')
            L = max(len(f), len(g)); f += [0] * (L - len(f)); g += [0] * (L - len(g))
            g = [g[i] ^ mul(c, f[i]) for i in range(L)][1:]
            delta += 1
        return all(pat[i:i + 2] in ('Sn', 'S0', '00') for i in range(0, 70, 2))
    gen = [generic(S) for err, S in shots]
    print('generic fraction', sum(gen) / len(gen))
    shots = [s for s, ok in zip(shots, gen) if ok]
    for err, S in shots:
        f = [1]; g = S[::-1]; bf = [0]; bg = [1]
        dg = lambda p: max([i for i, v in enumerate(p) if v] or [-1])
        good = True
        for step in range(70):
            if step % 2 == 0 and g[0] != 0:
                f, g, bf, bg = g, f, bg, bf
            c = mul(g[0], inv(f[0]))
            if c:
                lo[step] = max(lo[step], max(dg(f), dg(g)) + 1)
                hi[step] = min(hi[step], K - max(dg(bf), dg(bg)))
            L = max(len(f), len(g)); f += [0] * (L - len(f)); g += [0] * (L - len(g))
            g = [g[i] ^ mul(c, f[i]) for i in range(L)][1:]
            Lb = max(len(bf), len(bg)); bf += [0] * (Lb - len(bf)); bg += [0] * (Lb - len(bg))
            bg = [bg[i] ^ mul(c, bf[i]) for i in range(Lb)]
            bf = [0] + bf
    print('violations', [s for s in range(70) if lo[s] > hi[s]])
    P = lo
    print('P', P)
    nok = 0; nbad = 0
    for err, S in shots[: int(sys.argv[3])]:
        F, Gr, ok = forward(S, P)
        if not ok:
            nok += 1
            continue
        if decode(F, Gr) != err:
            nbad += 1
    print('inconsistent (non-generic)', nok, 'decode mismatches among consistent', nbad)


def cleanup(F, Gr, w):
    """Post-EEA rearrangement so sigma / Omega windows are clean (see design notes)."""
    F = F[:]; Gr = Gr[:]
    rot = lambda a, k: a[k % len(a):] + a[:k % len(a)]          # elements move to lower index by k
    Gr = rot(Gr, 2 * w - 36)
    for p in range(K - 35, 71):
        if p <= 70 - w:
            F[p], Gr[p] = Gr[p], F[p]
    Gr = rot(Gr, -(w - 1))
    sub = F[K - 35:]
    sub = sub[(35 - w) % 36:] + sub[:(35 - w) % 36]
    F[K - 35:] = sub
    return F, Gr


def chien(F, Gr):
    errs = {}
    sig = [F[K - k] for k in range(36)]
    om = Gr[:35]
    for j in range(4095):
        X = G.gpow(2, j)
        if poly_eval(sig, X) == 0:
            D = 0
            for k in range(1, 36, 2):
                D ^= mul(sig[k], G.gpow(X, k))
            errs[j] = mul(poly_eval(om, X), inv(D))
    return errs


def full_decode(S, P):
    F, Gr, ok = forward(S, P)
    w = 0
    bg = [F[K - j] for j in range(36)]
    # w = number of swaps = deg(bg) for generic shots
    return F, Gr, ok
