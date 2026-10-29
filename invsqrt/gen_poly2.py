"""Benchmark I, design 2: piecewise polynomial for 1/sqrt(x), products accumulated in short
registers (short carry chains), no Newton step.

    x = 2^e m, m in [1, 2):  1/sqrt(x) = 2^(L' - 10) g(L_0, m),  g = 2^(-p/2)/sqrt(m) in (1/2, 1]
    L = 21 - e (normalizing left shift, binary barrel shifter 16,8,4,2,1), p = 1 - L_0, L' = L >> 1.

    Degree d Horner on delta = m - m_seg (2^k segments), coefficients from one QROM:
        t_d = c_d;   P_j = T(t_{j+1} * delta)  (accumulated, guard bits);   t_j = c_j - P_j >> guard_j
    g = t_0;  y = g << L';  phase by y;  uncompute everything.

`model` reproduces the integer arithmetic of `build` exactly.
"""
from __future__ import annotations

import dataclasses
import math
from fractions import Fraction

import numpy as np
import kickmix as km

WIDTH = 22
MAX_X = 3 * (2**10 - 2) ** 2
MANT = WIDTH - 1
SHIFT_STAGES = (16, 8, 4, 2, 1)


@dataclasses.dataclass(frozen=True)
class Design:
    k: int = 7
    degree: int = 2
    g_bits: int = 27            # fractional bits of t_0 = g (= c_0)
    t_bits: tuple = (21, 12)    # fractional bits of t_1 .. t_d
    guard: tuple = (3, 2)       # guard bits of P_0 .. P_{d-1}
    dbits: tuple = (20, 12)     # delta bits used for P_0 .. P_{d-1} (MSB first)
    use_and: bool = True        # rows as temporary-AND + uncontrolled add

    @property
    def seg_shift(self) -> int:
        return MANT - self.k

    def out_bits(self, j: int) -> int:   # fractional bits of t_j
        return self.g_bits if j == 0 else self.t_bits[j - 1]


def _cheb_coeffs(des: Design):
    from numpy.polynomial import chebyshev as C
    h = 2.0 ** -des.k
    res = {}
    for p in (0, 1):
        for seg in range(1 << des.k):
            a = 1 + seg * h
            t = np.linspace(0, h, 3000)
            f = 2.0 ** (-p / 2) / np.sqrt(a + t)
            cf = C.chebfit(2 * t / h - 1, f, des.degree)
            poly = C.cheb2poly(cf)
            mono = np.zeros(des.degree + 1)
            for i, ci in enumerate(poly):
                for j in range(i + 1):
                    mono[j] += ci * math.comb(i, j) * (2 / h) ** j * (-1) ** (i - j)
            res[(p, seg)] = mono
    return res


def _rows(des: Design, j: int):
    """Rows of P_j: (xn bit, m) with delta bit weight 2^-m, MSB first."""
    out = []
    for i in range(des.dbits[j]):
        bit = des.seg_shift - 1 - i
        if bit < 0:
            break
        out.append((bit, MANT - bit))
    return out


def _row_shift(des: Design, j: int, m: int) -> int:
    """P_j has out_bits(j)+guard_j fractional bits; row adds t_{j+1} * 2^-m."""
    return des.out_bits(j) + des.guard[j] - des.out_bits(j + 1) - m


def fit_tables(des: Design, bias: bool = True) -> list[list[int]]:
    """tables[j][addr] = magnitude of c_j in units 2^-out_bits(j); addr = L_0 * 2^k + seg.

    Signs alternate: g = c0 - c1 d + c2 d^2 ...  The c_j for j < d are rounded and biased up
    by half the expected truncation loss of the rows feeding them."""
    co = _cheb_coeffs(des)
    n = 1 << des.k
    tables = [[0] * (2 * n) for _ in range(des.degree + 1)]
    for L0 in (0, 1):
        p = 1 - L0
        for seg in range(n):
            mono = co[(p, seg)]
            addr = L0 * n + seg
            for j in range(des.degree + 1):
                v = abs(mono[j]) * 2 ** des.out_bits(j)
                if bias and j < des.degree:
                    # P_j is floored: rows * 0.5 LSB(P_j) expected loss, then >> guard (floor, 0.5 LSB)
                    loss = len(_rows(des, j)) * 0.5 * 2.0 ** -des.guard[j] + 0.5
                    v -= loss  # t_j = c_j - P_j: P_j too small -> t_j too big -> lower c_j
                tables[j][addr] = int(round(v))
    return tables


def model(des: Design, xs: np.ndarray, tables) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.int64)
    e = np.floor(np.log2(xs)).astype(np.int64)
    e = np.where((1 << (e + 1)) <= xs, e + 1, e)
    e = np.where((1 << e) > xs, e - 1, e)
    L = MANT - e
    xn = xs << L
    L0 = L & 1
    Lp = L >> 1
    seg = (xn >> des.seg_shift) & ((1 << des.k) - 1)
    addr = L0 * (1 << des.k) + seg
    tabs = [np.array(t, dtype=np.int64)[addr] for t in tables]
    t = tabs[des.degree]
    for j in range(des.degree - 1, -1, -1):
        P = np.zeros_like(xs)
        for bit, m in _rows(des, j):
            sh = _row_shift(des, j, m)
            ctl = (xn >> bit) & 1
            P += ctl * (t << sh if sh >= 0 else t >> -sh)
        t = tabs[j] - (P >> des.guard[j])
        if (t < 0).any():
            raise OverflowError(f'negative t_{j}')
    if (t >= 1 << (des.g_bits + 1)).any():
        raise OverflowError('g overflow')
    return t << Lp


def rel_error(des: Design, tables, xs=None):
    if xs is None:
        xs = np.arange(1, MAX_X + 1, dtype=np.int64)
    y = model(des, xs, tables)
    exact = 1 / np.sqrt(xs.astype(np.float64))
    return np.abs(y / 2.0 ** (des.g_bits + 10) - exact) / exact


def widths(des: Design, tables):
    """Register widths (integer + fractional bits) of t_1..t_d and P_0..P_{d-1}."""
    xs = np.arange(1, MAX_X + 1, dtype=np.int64)
    wt = [max(int(max(tables[j])).bit_length(), 1) for j in range(des.degree + 1)]
    # P_j max: t_{j+1} max * delta_max in P units -> bound by table maxima (t_j <= c_j)
    wP = []
    for j in range(des.degree):
        tmax = max(tables[j + 1])
        tot = 0
        for bit, m in _rows(des, j):
            sh = _row_shift(des, j, m)
            tot += (tmax << sh) if sh >= 0 else (tmax >> -sh)
        wP.append(max(int(tot).bit_length(), 1))
    return wt, wP


# ----------------------------------------------------------------------------- circuit


def _multi_and(b, ctrls, target, inverse):
    n = len(ctrls)
    if n == 1:
        b.cx(ctrls[0], target)
        return
    if n == 2:
        (b.del_and if inverse else b.init_and)(ctrls[0], ctrls[1], target)
        return
    tmp = b.alloc_qubits(n - 2)
    ands, prev = [], ctrls[0]
    for i in range(1, n - 1):
        b.init_and(prev, ctrls[i], tmp[i - 1])
        ands.append((prev, ctrls[i], tmp[i - 1]))
        prev = tmp[i - 1]
    (b.del_and if inverse else b.init_and)(prev, ctrls[n - 1], target)
    for c1, c2, t in reversed(ands):
        b.del_and(c1, c2, t)
    b.free(tmp)


def build(des: Design, tables) -> km.Circuit:
    wt, wP = widths(des, tables)
    b = km.CircuitBuilder()
    x = b.create_quantum_register(WIDTH, name='x')
    d = des.degree
    F = des.g_bits + 10
    ylen = des.g_bits + 11
    y = b.alloc_qubits(ylen)
    flags = b.alloc_qubits(len(SHIFT_STAGES))
    g = y[:des.g_bits + 1]
    treg = [g] + [b.alloc_qubits(wt[j]) for j in range(1, d + 1)]
    Preg = [b.alloc_qubits(wP[j]) for j in range(d)]

    seg_q = x[des.seg_shift:MANT]
    address = km.array(list(seg_q) + [flags[-1]])   # msb = L_0
    packed = []
    for a in range(2 << des.k):
        v, off = 0, 0
        for j in range(d + 1):
            assert tables[j][a] < (1 << len(treg[j])), (j, a)
            v |= tables[j][a] << off
            off += len(treg[j])
        packed.append(v)
    lookup_target = km.array([q for r in treg for q in r])

    def normalize(inverse):
        st = list(zip(SHIFT_STAGES, flags))
        for s, f in (reversed(st) if inverse else st):
            top = x[WIDTH - s:]
            if inverse:
                b.right_rotate(x, s, control=f)
            b.x(top)
            _multi_and(b, top, f, inverse)
            b.x(top)
            if not inverse:
                b.left_rotate(x, s, control=f)

    def add_rows(j, inverse):
        """P_j +=/-= rows(t_{j+1} * delta)."""
        src, P = treg[j + 1], Preg[j]
        rows = _rows(des, j)
        for bit, m in (reversed(rows) if inverse else rows):
            sh = _row_shift(des, j, m)
            off_src, off_dst = (0, sh) if sh >= 0 else (-sh, 0)
            piece = src[off_src:]
            if off_dst + len(piece) > len(P):
                piece = piece[:len(P) - off_dst]
            if len(piece) == 0:
                continue
            tgt = P[off_dst:]
            ctl = x[bit]
            if des.use_and:
                tmp = b.alloc_qubits(len(piece))
                b.init_and(ctl, piece, tmp)
                (b.isub if inverse else b.iadd)(tmp, target=tgt)
                b.del_and(ctl, piece, tmp)
                b.free(tmp)
            else:
                (b.isub if inverse else b.iadd)(piece, target=tgt, control=ctl)

    def horner(inverse):
        js = list(range(d - 1, -1, -1))
        for j in (js[::-1] if inverse else js):
            P, dst, gb = Preg[j], treg[j], des.guard[j]
            if not inverse:
                add_rows(j, False)
                b.isub(P[gb:], target=dst)
            else:
                b.iadd(P[gb:], target=dst)
                add_rows(j, True)

    def denormalize(inverse):
        st = list(zip(SHIFT_STAGES[:-1], flags[:-1]))
        for s, f in (reversed(st) if inverse else st):
            (b.right_rotate if inverse else b.left_rotate)(y, s // 2, control=f)

    with b.mark('normalize'):
        normalize(False)
    with b.mark('lookup'):
        b.init_lookup(table=packed, address=address, target=lookup_target)
    with b.mark('horner'):
        horner(False)
    with b.mark('denormalize'):
        denormalize(False)
    with b.mark('phase'):
        for j in range(ylen):
            ex = Fraction(2) ** (j - F + 1)
            if ex % 2 != 0:
                b.z_pow(y[j], ex)
    with b.mark('undenormalize'):
        denormalize(True)
    with b.mark('unhorner'):
        horner(True)
    with b.mark('unlookup'):
        b.del_lookup(table=packed, address=address, target=lookup_target)
    with b.mark('unnormalize'):
        normalize(True)
    for r in [y, flags] + treg[1:] + Preg:
        b.free(r)
    return b.finish_circuit()


def estimate(des: Design, tables=None, verbose=False):
    """(score estimate, details) using the verifier's cost model (rotation eps from margin)."""
    if tables is None:
        tables = fit_tables(des)
    err = rel_error(des, tables)
    worst = err.max()
    if worst >= 2.0**-24:
        return math.inf, dict(err=math.log2(worst))
    c = build(des, tables)
    st = c.operation_statistics()
    xs = np.arange(1, MAX_X + 1, dtype=np.float64)
    margin = ((2.0**-24 - err) / np.sqrt(xs)).min()          # turns
    n_rot = st.num_z_pow(min_clifford_hierarchy_level=4)
    eps = math.pi * margin / n_rot
    tcount = st.num_t_like + n_rot * (0.56 * math.log2(1 / eps) + 5.3)
    C = st.num_toffoli_like + tcount / 4
    return c.num_qubits * C, dict(err=math.log2(worst), Q=c.num_qubits, tof=st.num_toffoli_like,
                                   n_rot=n_rot, C=C)


if __name__ == '__main__':
    s, info = estimate(Design())
    print(f'{s:,.0f}', info)
