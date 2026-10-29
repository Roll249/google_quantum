"""Classical model: Bernstein-Yang divsteps with explicit Bezout (G-coefficients) for the RS key equation."""
import sys
import numpy as np
sys.path.insert(0, 'dqi')
import gfalg as G

exp, log = G.tables()
n, m, ell = 70, 4095, 35


def mul(a, b): return G.gmul(a, b)
def inv(a): return G.ginv(a) if a else 0


def syndromes(err):
    S = [0] * n
    for j, v in err.items():
        for i in range(n):
            S[i] ^= mul(v, G.gpow(2, ((i + 1) * j) % 4095))
    return S


def poly_eval(p, x):
    r = 0
    for c in reversed(p):
        r = mul(r, x) ^ c
    return r


def run(S, N=2 * ell, trace=False):
    f = [1] + [0] * (n - 1)
    g = S[::-1]
    delta = 1
    bf, bg = [0], [1]
    log_ = []
    for step in range(N):
        sw = delta > 0 and g[0] != 0
        if sw:
            f, g, bf, bg, delta = g, f, bg, bf, -delta
        c = mul(g[0], inv(f[0]))
        L = max(len(bg), len(bf))
        bg = [(bg[i] if i < len(bg) else 0) ^ mul(c, bf[i] if i < len(bf) else 0) for i in range(L)]
        g = [g[i] ^ mul(c, f[i]) for i in range(len(g))]
        assert g[0] == 0
        g = g[1:]
        f = f[:len(g)]
        bf = [0] + bf
        delta += 1
        dg = lambda p: max([i for i, v in enumerate(p) if v] or [-1])
        log_.append((step, sw, delta, dg(bf), dg(bg), len(f), c != 0))
    return f, g, bf, bg, delta, log_


if __name__ == '__main__':
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    for trial in range(6):
        w = [35, 34, 31, 20, 1, 0][trial]
        pos = rng.choice(m, w, replace=False)
        err = {int(p): int(rng.integers(1, 4096)) for p in pos}
        S = syndromes(err)
        f, g, bf, bg, delta, lg = run(S)
        res = {}
        for name, p in (('bf', bf), ('bg', bg)):
            for rev in (False, True):
                q = p[::-1] if rev else p
                ok = all(poly_eval(q, G.ginv(G.gpow(2, j))) == 0 for j in err) if err else None
                res[(name, rev)] = ok
        dgs = lambda p: max([i for i, v in enumerate(p) if v] or [-1])
        print(f'w={w} delta={delta} deg bf={dgs(bf)} deg bg={dgs(bg)} roots:', res)
        if trial == 0:
            for row in lg[:12]:
                print('   ', row)

