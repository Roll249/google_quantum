"""Benchmark I (1/sqrt(x) phasing): piecewise-polynomial circuit without a Newton step.

    x = 2^e m, m in [1, 2), e in [0, 21]            (x in [1, max_x], 22-bit register)
    1/sqrt(x) = 2^(-10) 2^L' g(p, m),  g = 2^(-p/2) / sqrt(m) in (1/2, 1]

with L = 21 - e the left shift that normalizes x (binary barrel shifter, stages 16..1),
p = 1 - L_0 the parity of e and L' = L >> 1.  g is a degree-d polynomial in
delta = m - m_seg on 2^k segments of [1, 2), coefficients (indexed by p and the segment)
from one QROM lookup.  Horner with truncated rows:

    t_{d-1} = c_{d-1} - T(c_d delta),  ...,  g = c_0 - T(t_1 delta)   (alternating signs)

then y = g << L' (barrel shift), one Z_POW per bit of y, and everything uncomputed in
reverse.  `model` reproduces the circuit arithmetic exactly in integers, so the worst
relative error over the whole domain is known before building the circuit.
"""
from __future__ import annotations

import dataclasses
import math
from fractions import Fraction

import numpy as np
import kickmix as km

WIDTH = 22
MAX_X = 3 * (2**10 - 2) ** 2
MANT = WIDTH - 1          # m = xn / 2^21 after normalization (top bit set)
SHIFT_STAGES = (16, 8, 4, 2, 1)


@dataclasses.dataclass(frozen=True)
class Design:
    k: int = 7                 # 2^k segments of [1, 2)
    degree: int = 2
    g_bits: int = 27           # fractional bits of g (and of c_0)
    coef_bits: tuple = (21, 13)  # fractional bits of t_1..t_{d} (c_1..c_d registers), one per degree >= 1
    delta_bits: tuple = (21, 13)  # bits of delta used by the rows feeding t_{j-1}, j = 1..d (MSB first)

    @property
    def seg_shift(self) -> int:  # delta bits live below this bit of xn
        return MANT - self.k


def fit_tables(des: Design) -> list[list[int]]:
    """coef[j][addr] (non-negative magnitudes, alternating signs implied), addr = p * 2^k + seg."""
    from numpy.polynomial import chebyshev as C
    h = 2.0**-des.k
    tables = [[0] * (2 << des.k) for _ in range(des.degree + 1)]
    for p in (0, 1):
        for seg in range(1 << des.k):
            a = 1 + seg * h
            t = np.linspace(0, h, 2000)
            f = 2.0 ** (-p / 2) / np.sqrt(a + t)
            # minimax-ish: Chebyshev fit on [0, h], converted to monomial in delta
            cf = C.chebfit(2 * t / h - 1, f, des.degree)
            poly = C.cheb2poly(cf)  # in u = 2 delta / h - 1
            # expand in delta: u = (2/h) delta - 1
            mono = np.zeros(des.degree + 1)
            for i, ci in enumerate(poly):
                # (2/h d - 1)^i
                for j in range(i + 1):
                    mono[j] += ci * math.comb(i, j) * (2 / h) ** j * (-1) ** (i - j)
            addr = p * (1 << des.k) + seg
            for j in range(des.degree + 1):
                mag = abs(mono[j])
                bits = des.g_bits if j == 0 else des.coef_bits[j - 1]
                tables[j][addr] = int(round(mag * 2**bits))
    return tables


def _rows(delta_bits_avail: int, used: int, seg_shift: int):
    """(xn bit index, m) for delta bits with weight 2^-m in m-units (m = 1..), MSB first."""
    out = []
    for i in range(used):
        bit = seg_shift - 1 - i
        if bit < 0:
            break
        out.append((bit, MANT - bit))  # xn bit b has weight 2^(b - 21) in m
    return out


def model(des: Design, xs: np.ndarray, tables=None) -> np.ndarray:
    """Exact integer model: y with F = g_bits + 10 fractional bits (turns) per x."""
    if tables is None:
        tables = fit_tables(des)
    xs = np.asarray(xs, dtype=np.int64)
    e = np.floor(np.log2(xs)).astype(np.int64)
    # guard against float log2 errors
    e = np.where((1 << (e + 1)) <= xs, e + 1, e)
    e = np.where((1 << e) > xs, e - 1, e)
    L = MANT - e
    xn = xs << L
    p = 1 - (L & 1)
    Lp = L >> 1
    seg = (xn >> des.seg_shift) & ((1 << des.k) - 1)
    addr = p * (1 << des.k) + seg
    tabs = [np.array(t, dtype=np.int64)[addr] for t in tables]
    d = des.degree
    # Horner from the top: acc_j has coef_bits[j-1] (j>=1) or g_bits (j=0) fractional bits
    acc = tabs[d]
    acc_bits = des.coef_bits[d - 1] if d >= 1 else des.g_bits
    for j in range(d - 1, -1, -1):
        out_bits = des.g_bits if j == 0 else des.coef_bits[j - 1]
        total = np.zeros_like(xs)
        for bit, m in _rows(None, des.delta_bits[j], des.seg_shift):
            # row: control = xn bit (weight 2^-m), adds floor(acc * 2^(out_bits - acc_bits - m))
            sh = out_bits - acc_bits - m
            ctl = (xn >> bit) & 1
            total += ctl * (acc << sh if sh >= 0 else acc >> -sh)
        # sign: g = c0 - c1 d + c2 d^2 - ...  => t_{j} = c_j - t_{j+1} d
        acc = tabs[j] - total
        acc_bits = out_bits
        if (acc < 0).any():
            raise OverflowError(f'negative Horner intermediate at degree {j}')
    g = acc  # g_bits fractional
    return g << Lp  # y with g_bits + 10 fractional bits: y / 2^(g_bits+10) = 1/sqrt(x)


def rel_error(des: Design, xs=None, tables=None):
    if xs is None:
        xs = np.arange(1, MAX_X + 1, dtype=np.int64)
    y = model(des, xs, tables)
    F = des.g_bits + 10
    exact = 1 / np.sqrt(xs.astype(np.float64))
    return np.abs(y / 2.0**F - exact) / exact


# ----------------------------------------------------------------------------- circuit


def build(des: Design, tables=None) -> km.Circuit:
    if tables is None:
        tables = fit_tables(des)
    b = km.CircuitBuilder()
    x = b.create_quantum_register(WIDTH, name='x')
    d = des.degree
    F = des.g_bits + 10
    ylen = des.g_bits + 11          # g (g_bits frac + 1 int bit) shifted left by up to 10
    y = b.alloc_qubits(ylen)
    flags = b.alloc_qubits(len(SHIFT_STAGES))
    coef_regs = [b.alloc_qubits(des.coef_bits[j - 1] + 1) for j in range(1, d + 1)]
    g = y[:des.g_bits + 1]
    addr_bits = des.k + 1

    def normalize(inverse: bool):
        stages = list(zip(SHIFT_STAGES, flags))
        if not inverse:
            for s, f in stages:
                top = x[WIDTH - s:]
                # f := all(top == 0)
                b.x(top)
                _multi_and(b, top, f, inverse=False)
                b.x(top)
                b.left_rotate(x, s, control=f)
        else:
            for s, f in reversed(stages):
                b.right_rotate(x, s, control=f)
                top = x[WIDTH - s:]
                b.x(top)
                _multi_and(b, top, f, inverse=True)
                b.x(top)

    # address = (seg bits of xn, parity p = NOT flag of stage 1) little-endian: seg then p
    seg_q = x[des.seg_shift:MANT]
    p_q = flags[-1]   # flag of stage 1 = L_0;  p = 1 - L_0  -> table indexed by L_0 instead
    address = km.array(list(seg_q) + [p_q])
    # reorder tables: address msb = L_0, p = 1 - L_0
    def reindexed(t):
        n = 1 << des.k
        return [t[(1 - (a >> des.k)) * n + (a & (n - 1))] for a in range(2 * n)]
    tabs = [reindexed(t) for t in tables]
    lookup_targets = [g] + coef_regs  # c0 into g, c_j into coef_regs[j-1]
    widths = [len(r) for r in lookup_targets]
    packed = []
    for a in range(2 << des.k):
        v, off = 0, 0
        for j, w in enumerate(widths):
            assert tabs[j][a] < (1 << w)
            v |= tabs[j][a] << off
            off += w
        packed.append(v)
    target_all = km.array([q for r in lookup_targets for q in r])

    def horner(inverse: bool):
        steps = list(range(d - 1, -1, -1))
        if inverse:
            steps = steps[::-1]
        for j in steps:
            src = coef_regs[j]  # t_{j+1}
            src_bits = des.coef_bits[j]
            dst = g if j == 0 else coef_regs[j - 1]
            out_bits = des.g_bits if j == 0 else des.coef_bits[j - 1]
            for bit, m in _rows(None, des.delta_bits[j], des.seg_shift):
                sh = out_bits - src_bits - m
                if sh >= 0:
                    off_src, off_dst = 0, sh
                else:
                    off_src, off_dst = -sh, 0
                piece = src[off_src:]
                if off_dst + len(piece) > len(dst):
                    piece = piece[:len(dst) - off_dst]
                if len(piece) == 0:
                    continue
                tgt = dst[off_dst:]
                if inverse:
                    b.iadd(piece, target=tgt, control=x[bit])
                else:
                    b.isub(piece, target=tgt, control=x[bit])

    def denormalize(inverse: bool):
        stages = list(zip(SHIFT_STAGES[:-1], flags[:-1]))  # L' = L >> 1: shifts 8,4,2,1
        if not inverse:
            for s, f in stages:
                b.left_rotate(y, s // 2, control=f)
        else:
            for s, f in reversed(stages):
                b.right_rotate(y, s // 2, control=f)

    with b.mark('normalize'):
        normalize(False)
    with b.mark('lookup'):
        b.init_lookup(table=packed, address=address, target=target_all)
    with b.mark('horner'):
        horner(False)
    with b.mark('denormalize'):
        denormalize(False)
    with b.mark('phase'):
        for j in range(ylen):
            ex = Fraction(2) ** (j - F + 1)  # half turns: 2^(j-F) turns
            if ex % 2 != 0:
                b.z_pow(y[j], ex)
    with b.mark('undenormalize'):
        denormalize(True)
    with b.mark('unhorner'):
        horner(True)
    with b.mark('unlookup'):
        b.del_lookup(table=packed, address=address, target=target_all)
    with b.mark('unnormalize'):
        normalize(True)
    for r in [y, flags] + coef_regs:
        b.free(r)
    return b.finish_circuit()


def _multi_and(b: km.CircuitBuilder, ctrls: km.array, target: km.q, inverse: bool):
    """target ^= AND(ctrls) (target clean before compute); compute-uncompute pair."""
    n = len(ctrls)
    if n == 1:
        b.cx(ctrls[0], target)
        return
    if n == 2:
        if inverse:
            b.del_and(ctrls[0], ctrls[1], target)
        else:
            b.init_and(ctrls[0], ctrls[1], target)
        return
    # chain of temporary ANDs
    tmp = b.alloc_qubits(n - 2)
    chain = [ctrls[0]] + list(tmp)
    ands = []
    prev = ctrls[0]
    for i in range(1, n - 1):
        b.init_and(prev, ctrls[i], tmp[i - 1])
        ands.append((prev, ctrls[i], tmp[i - 1]))
        prev = tmp[i - 1]
    if inverse:
        b.del_and(prev, ctrls[n - 1], target)
    else:
        b.init_and(prev, ctrls[n - 1], target)
    for c1, c2, t in reversed(ands):
        b.del_and(c1, c2, t)
    b.free(tmp)


if __name__ == '__main__':
    import sys
    des = Design()
    err = rel_error(des)
    print(f'max rel err 2^{math.log2(err.max()):.2f} at x={err.argmax()+1}')
    c = build(des)
    st = c.operation_statistics()
    print('qubits', c.num_qubits, 'toffoli', st.num_toffoli_like, 'zpow', st.num_z_pow())
