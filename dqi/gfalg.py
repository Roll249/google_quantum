"""Bilinear algorithms over GF(2^12) (modulus 0x1053, little-endian polynomial basis).

A bilinear algorithm for z = x*y is a list of products p_k = (U_k . x)(V_k . y) over GF(2)
plus an output matrix W with z = W p.  Built from the tower GF(2^12) = GF(8)[t]/(min poly of
theta, degree 4): evaluate the degree-3 GF(8)-polynomials at 7 points (6 finite + infinity),
multiply each pair of GF(8) values with the 6-AND Karatsuba formula, interpolate.  W is solved
by GF(2) linear algebra and the whole algorithm is checked exhaustively.
"""
from __future__ import annotations

import functools
import numpy as np

B = 12
MOD = 0x1053
ORDER = (1 << B) - 1


def gmul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> B:
            a ^= MOD
    return r


@functools.lru_cache(None)
def tables():
    exp = [0] * (2 * ORDER)
    x = 1
    for i in range(ORDER):
        exp[i] = x
        x = gmul(x, 2)
    for i in range(ORDER, 2 * ORDER):
        exp[i] = exp[i - ORDER]
    log = [0] * (1 << B)
    for i in range(ORDER):
        log[exp[i]] = i
    return exp, log


def gpow(a: int, e: int) -> int:
    if a == 0:
        return 0 if e else 1
    exp, log = tables()
    return exp[(log[a] * e) % ORDER]


def ginv(a: int) -> int:
    return gpow(a, ORDER - 1)


# ---------------------------------------------------------------- GF(2) linear algebra


def gf2_solve_rows(A: np.ndarray, Y: np.ndarray):
    """Find X with X @ A = Y (mod 2) for A (r x s), Y (t x s); returns X (t x r) or None."""
    r, s = A.shape
    M = np.concatenate([A.T.copy(), Y.T.copy()], axis=1).astype(np.uint8) % 2  # s x (r + t)
    piv = []
    row = 0
    for col in range(r):
        p = None
        for i in range(row, s):
            if M[i, col]:
                p = i
                break
        if p is None:
            continue
        M[[row, p]] = M[[p, row]]
        for i in range(s):
            if i != row and M[i, col]:
                M[i] ^= M[row]
        piv.append(col)
        row += 1
    # consistency
    for i in range(row, s):
        if M[i, r:].any():
            return None
    X = np.zeros((Y.shape[0], r), dtype=np.uint8)
    for k, col in enumerate(piv):
        X[:, col] = M[k, r:]
    return X


def bits(v: int, n: int = B) -> np.ndarray:
    return np.array([(v >> i) & 1 for i in range(n)], dtype=np.uint8)


def mat_of_linear(f, n_in: int = B, n_out: int = B) -> np.ndarray:
    """Matrix (n_out x n_in) of a GF(2)-linear map on ints."""
    return np.array([bits(f(1 << i), n_out) for i in range(n_in)], dtype=np.uint8).T


# ---------------------------------------------------------------- tower construction





def build_mul_algorithm():
    """Returns (U, V, W): U, V are (K x 12) linear forms, W is (12 x K)."""
    beta = gpow(2, ORDER // 7)          # generator of GF(8)*
    theta = 2                            # alpha generates GF(2^12) over GF(8)
    basis = [gmul(gpow(beta, s), gpow(theta, i)) for i in range(4) for s in range(3)]  # index 3*i + s
    Bm = np.array([bits(v) for v in basis], dtype=np.uint8).T      # 12 x 12: coords -> bits
    Binv = gf2_solve_rows(Bm, np.eye(B, dtype=np.uint8))          # X @ Bm = I
    assert Binv is not None
    # coordinate map: c = Bm^{-1} x ; with X @ Bm = I, X = Bm^{-1}
    C = Binv  # 12 x 12, rows: coord index 3*i+s
    # GF(8) values as 3-bit vectors over basis (1, beta, beta^2); value of A(y) at point y=pt:
    # A(pt) = sum_i a_i pt^i, a_i = sum_s c[3i+s] beta^s.  Multiplication by pt^i on GF(8)
    # in basis (1,beta,beta^2) is a 3x3 matrix.
    g8 = [gpow(beta, s) for s in range(3)]
    g8m = np.array([bits(v) for v in g8], dtype=np.uint8).T  # 12 x 3
    g8inv = gf2_solve_rows(g8m, np.eye(3, dtype=np.uint8)[:, :3].T @ np.zeros((3, 3), np.uint8) + np.eye(3, dtype=np.uint8)) if False else None

    def to8(v):  # GF(8) element (as GF(2^12) int) -> 3 bits over (1, beta, beta^2)
        for k in range(8):
            if (k & 1) * g8[0] ^ ((k >> 1) & 1) * g8[1] ^ ((k >> 2) & 1) * g8[2] == v:
                return bits(k, 3)
        raise ValueError(v)

    pts = [0] + [gpow(beta, j) for j in range(5)] + [None]  # 6 finite points + infinity
    U, V = [], []
    for pt in pts:
        # E: 3 x 12 matrix giving GF(8) coords of A(pt) from x bits
        E = np.zeros((3, B), dtype=np.uint8)
        for i in range(4):
            if pt is None:
                if i != 3:
                    continue
                f = 1
            else:
                f = gpow(pt, i) if (pt or i == 0) else 0
                if pt == 0:
                    f = 1 if i == 0 else 0
            if f == 0:
                continue
            for s in range(3):
                # contribution of coord (i,s): beta^s * f, as GF(8) vector
                E[:, :] ^= np.outer(to8(gmul(g8[s], f)), C[3 * i + s]) % 2
        # Karatsuba 6 products over the 3 GF(8) coordinates
        e0, e1, e2 = E
        for lf in (e0, e1, e2, e0 ^ e1, e0 ^ e2, e1 ^ e2):
            U.append(lf)
            V.append(lf)
    U = np.array(U, dtype=np.uint8)
    V = np.array(V, dtype=np.uint8)
    W = solve_output(U, V)
    return U, V, W


def products(U, V, xs, ys):
    """xs, ys: int arrays -> (K, N) product bits."""
    xb = ((xs[None, :] >> np.arange(B)[:, None]) & 1).astype(np.uint8)
    yb = ((ys[None, :] >> np.arange(B)[:, None]) & 1).astype(np.uint8)
    return ((U @ xb) % 2) & ((V @ yb) % 2)


def gmul_vec(xs, ys):
    exp, log = tables()
    exp = np.array(exp); log = np.array(log)
    out = exp[(log[xs] + log[ys]) % ORDER]
    return np.where((xs == 0) | (ys == 0), 0, out)


def solve_output(U, V, func=None, n_out=B, samples=4000, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, 1 << B, samples)
    ys = rng.integers(0, 1 << B, samples)
    P = products(U, V, xs, ys)
    z = gmul_vec(xs, ys) if func is None else func(xs, ys)
    Z = ((z[None, :] >> np.arange(n_out)[:, None]) & 1).astype(np.uint8)
    W = gf2_solve_rows(P, Z)
    if W is None:
        raise ValueError('products do not span the output')
    return W


def check(U, V, W, func=None, n_out=B):
    xs = np.repeat(np.arange(1 << B), 1)
    for y0 in range(0, 1 << B, 256):
        ys = np.arange(y0, y0 + 256)
        X, Y = np.meshgrid(xs, ys)
        X = X.ravel(); Y = Y.ravel()
        P = products(U, V, X, Y)
        Z = (W.astype(np.int64) @ P) % 2
        z = gmul_vec(X, Y) if func is None else func(X, Y)
        Zt = ((z[None, :] >> np.arange(n_out)[:, None]) & 1)
        if (Z != Zt).any():
            return False
    return True


if __name__ == '__main__':
    U, V, W = build_mul_algorithm()
    print('products', len(U), 'exhaustive ok', check(U, V, W))


# ---------------------------------------------------------------- tower GF(8) < GF(64) < GF(2^12)
# Tower coordinates: index i + 3 j + 6 k  <->  beta^i s^j theta^k  (i<3, j<2, k<2).


@functools.lru_cache(None)
def tower():
    beta = gpow(2, ORDER // 7)
    s = gpow(2, ORDER // 63)
    theta = 2
    T = [gmul(gmul(gpow(beta, i), gpow(s, j)), gpow(theta, k))
         for k in range(2) for j in range(2) for i in range(3)]
    Tm = np.array([bits(v) for v in T], dtype=np.uint8).T          # bits = Tm @ coords
    Tinv = gf2_solve_rows(Tm, np.eye(B, dtype=np.uint8))           # coords = Tinv @ bits
    assert Tinv is not None and ((Tinv.astype(int) @ Tm) % 2 == np.eye(B)).all()
    return T, Tm, Tinv


ENC_POLY = tuple(1 << t for t in range(B))


def enc_tower(width: int):
    """Encoding of a register holding a GF(8) (3), GF(64) (6) or GF(2^12) (12) element in tower coords."""
    T, _, _ = tower()
    return tuple(T[:width])


def coords(enc) -> np.ndarray:
    """(12 x width) matrix: tower coordinates of the value held by a register with encoding enc."""
    _, _, Tinv = tower()
    E = np.array([bits(v) for v in enc], dtype=np.uint8).T             # 12 x width
    return (Tinv.astype(int) @ E) % 2


def _mul8(A, Bm):
    """Karatsuba GF(8) product forms for coordinate triples A (3 x na), Bm (3 x nb)."""
    U, V = [], []
    for sel in ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2)):
        U.append(np.bitwise_xor.reduce(A[list(sel)], axis=0))
        V.append(np.bitwise_xor.reduce(Bm[list(sel)], axis=0))
    return U, V


def _mul64(A, Bm):
    """A, Bm: (6 x n) GF(64) coords (rows i + 3 j) -> 18 product forms."""
    a0, a1, b0, b1 = A[:3], A[3:6], Bm[:3], Bm[3:6]
    U, V = [], []
    for x, y in ((a0, b0), (a1, b1), (a0 ^ a1, b0 ^ b1)):
        u, v = _mul8(x, y)
        U += u
        V += v
    return U, V


def value_of(enc, regbits: np.ndarray) -> np.ndarray:
    """regbits (width x N) -> GF(2^12) ints."""
    out = np.zeros(regbits.shape[1], dtype=np.int64)
    for t, v in enumerate(enc):
        out ^= np.where(regbits[t] == 1, v, 0)
    return out


def solve_map(U, V, f, enc_x, enc_y, enc_z, quadratic: bool, samples=3000, seed=1):
    """Solve W (and L for quadratic maps) with z = W p (+ L x) for the given product forms."""
    rng = np.random.default_rng(seed)
    U = np.array(U, dtype=np.uint8) % 2
    V = np.array(V, dtype=np.uint8) % 2
    xb = rng.integers(0, 2, (len(enc_x), samples)).astype(np.uint8)
    yb = xb if quadratic else rng.integers(0, 2, (len(enc_y), samples)).astype(np.uint8)
    xv = value_of(enc_x, xb)
    zv = f(xv) if quadratic else f(xv, value_of(enc_y, yb))
    Ez = np.array([bits(v) for v in enc_z], dtype=np.uint8).T          # 12 x wz
    Zb = ((zv[None, :] >> np.arange(B)[:, None]) & 1).astype(np.uint8)  # 12 x N
    R = gf2_solve_rows(Ez, np.eye(len(enc_z), dtype=np.uint8))           # R @ Ez = I
    assert R is not None, 'output encoding not independent'
    zr = (R.astype(int) @ Zb) % 2
    assert ((Ez.astype(int) @ zr) % 2 == Zb).all(), 'value outside output span'
    P = ((U.astype(int) @ xb) % 2) & ((V.astype(int) @ yb) % 2)
    A = np.concatenate([P, xb], axis=0) if quadratic else P
    X = gf2_solve_rows(A.astype(np.uint8), zr.astype(np.uint8))
    if X is None:
        raise ValueError('product forms do not span the map')
    W = X[:, :len(U)]
    L = X[:, len(U):] if quadratic else None
    return U, V, W, L


def _vec(f):
    return np.vectorize(f, otypes=[np.int64])


@functools.lru_cache(None)
def algorithms():
    """name -> (U, V, W, L, enc_x, enc_y, enc_z) for the decoder's field operations."""
    e3, e6 = enc_tower(3), enc_tower(6)
    out = {}
    U, V, W = build_mul_algorithm()
    out['mul'] = (U, V, W, None, ENC_POLY, ENC_POLY, ENC_POLY)
    cx = coords(ENC_POLY)
    U, V = _mul64(cx[:6], cx[6:])
    out['norm'] = (*solve_map(U, V, _vec(lambda x: gpow(x, 65)), ENC_POLY, None, e6, True), ENC_POLY, None, e6)
    c6 = coords(e6)
    U, V = _mul8(c6[:3], c6[3:6])
    out['norm8'] = (*solve_map(U, V, _vec(lambda x: gpow(x, 9)), e6, None, e3, True), e6, None, e3)
    I3 = np.eye(3, dtype=np.uint8)
    U = [I3[0], I3[0], I3[1]]
    V = [I3[1], I3[2], I3[2]]
    out['inv8'] = (*solve_map(U, V, _vec(lambda x: gpow(x, 6)), e3, None, e3, True), e3, None, e3)
    one = (1,)
    U = [np.array([1], np.uint8)] * 3
    V = list(I3)
    out['gate8'] = (*solve_map(U, V, _vec(lambda z, m: m if z else 0), one, e3, e3, False), one, e3, e3)
    c3 = coords(e3)
    U, V = [], []
    for j in range(2):
        u, v = _mul8(c6[3 * j:3 * j + 3], c3[:3])
        U += u; V += v
    out['ninv'] = (*solve_map(U, V, _vec(lambda n, m: gmul(gpow(n, 8), m)), e6, e3, e6, False), e6, e3, e6)
    U, V = [], []
    for k in range(2):
        u, v = _mul64(cx[6 * k:6 * k + 6], c6[:6])
        U += u; V += v
    out['xinv'] = (*solve_map(U, V, _vec(lambda x, n: gmul(gpow(x, 64), n)), ENC_POLY, e6, ENC_POLY, False), ENC_POLY, e6, ENC_POLY)
    return out


if __name__ == '__main__':
    algs = algorithms()
    for k, (U, V, W, L, *_) in algs.items():
        print(k, 'products', len(U))
    # end-to-end inverse check over all nonzero x
    xs = np.arange(1, 1 << B)
