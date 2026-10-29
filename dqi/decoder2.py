"""Benchmark II decoder v2: static-layout divstep EEA + cleaned Chien/Forney windows.

Key-equation solver: Bernstein-Yang divsteps on (f, g) = (1, rev S) with explicit Bezout
polynomials, run as 35 fixed "pairs" (conditional swap step + plain step).  Arrays

    F = (f from index 0 up,  bg reversed from the top: bg_j at F[K - j])
    G = (g from index 0 up,  bf reversed from the top: bf_j at G[K - j])

make every shift a free relabel (G <- G[1:] + [G[0]] drops the eliminated g_0 and appends
bf's new zero constant term), so each step is: one conditional array swap (pair steps only),
c = g_0 / f_0, and N ungated multiply-accumulates (g ^= c f on the low part, bg ^= c bf on the
high part, split at a static index P(step)).  The two quotient coefficients of a pair are
recovered from Bezout low coefficients at the pair end and erased:

    c' = bg_0 / bf_2,   c = (bg_1 + c' bf_3) / bf_2,   s = [bf_2 != 0].

Shots whose divstep pattern is not made of the pairs "Sn", "S0", "00" (a degree-2 quotient,
~0.7% of the test distribution) are decoded wrongly; the verifier prices this through R(eps).

After the EEA, sigma (bg) and Omega (g) are moved into clean windows with data-dependent
rotations by w = deg sigma (counted from the swaps), then Chien search + Forney evaluate

    e_j = Omega~(X) / (X sigma~'(X))   at X = alpha^j  (common offset X^(35 - w) cancels)

and kick back (-1)^{E_j . e_j}.  Everything is then uncomputed in reverse.
"""
from __future__ import annotations

import sys
from typing import Sequence

import kickmix as km

import gfq
from decoder import _init_is_zero, _del_is_zero

N = 73
K = N - 1
NSYN = 70
PAIRS = 35
DEG = 12
WBITS = 6


def split(step: int) -> int:
    return 70 if step == 0 else 70 - (step - 1) // 2


MUL = None


def _mul():
    global MUL
    if MUL is None:
        MUL = gfq.alg('mul')
    return MUL


def _nonzero_flag(b: km.CircuitBuilder, reg: km.array, target) -> None:
    """target ^= (reg != 0) (11 Toffolis, ladder cleaned for free)."""
    z = b.alloc_qubits(1)[0]
    lad = _init_is_zero(b, reg, z)
    b.cx(z, target)
    _del_is_zero(b, reg, z, lad)
    b.free(km.array([z]))
    b.x(target)


def _cswap_arrays(b, s, F, G):
    for i in range(N):
        b.cswap(s, F[i], G[i])


def _updates(b, c, F, G, step):
    p = split(step)
    for i in range(N):
        if i < p:
            gfq.mul_acc(b, c, F[i], G[i])
        else:
            gfq.mul_acc(b, c, G[i], F[i])


def pair_forward(b, F, G, w, k):
    s = b.alloc_qubits(1)[0]
    _nonzero_flag(b, G[0], s)
    _cswap_arrays(b, s, F, G)
    b.iadd(1, target=w, control=s)
    inv0, st0 = gfq.init_inverse(b, F[0])
    cs = []
    for half in range(2):
        c = gfq.init(b, _mul(), G[0], inv0)
        _updates(b, c, F, G, 2 * k + half)
        G = G[1:] + [G[0]]
        cs.append(c)
    gfq.del_inverse(b, st0)
    # pair end: erase the quotient coefficients from Bezout low coefficients
    iv, st = gfq.init_inverse(b, G[K - 2])
    t = gfq.init(b, _mul(), cs[0], G[K - 3])
    b.cx(F[K - 1], t)
    gfq.mul_acc(b, t, iv, cs[1])
    b.cx(F[K - 1], t)
    gfq.delete(b, _mul(), cs[0], G[K - 3], t)
    gfq.mul_acc(b, F[K], iv, cs[0])
    gfq.del_inverse(b, st)
    b.free(cs[0] + cs[1])
    _nonzero_flag(b, G[K - 2], s)
    b.free(km.array([s]))
    return F, G


def pair_reverse(b, F, G, w, k):
    s = b.alloc_qubits(1)[0]
    _nonzero_flag(b, G[K - 2], s)
    iv, st = gfq.init_inverse(b, G[K - 2])
    c1 = gfq.init(b, _mul(), F[K], iv)
    t = gfq.init(b, _mul(), c1, G[K - 3])
    b.cx(F[K - 1], t)
    c2 = gfq.init(b, _mul(), t, iv)
    b.cx(F[K - 1], t)
    gfq.delete(b, _mul(), c1, G[K - 3], t)
    gfq.del_inverse(b, st)
    inv0, st0 = gfq.init_inverse(b, F[0])
    for half, c in ((1, c2), (0, c1)):
        G = [G[-1]] + G[:-1]
        _updates(b, c, F, G, 2 * k + half)
        gfq.mul_acc(b, G[0], inv0, c)
        b.free(c)
    gfq.del_inverse(b, st0)
    b.isub(1, target=w, control=s)
    _cswap_arrays(b, s, F, G)
    _nonzero_flag(b, G[0], s)
    b.free(km.array([s]))
    return F, G


def _rotate(b, regs, amount, left: bool):
    """Data-dependent cyclic rotation of a list of registers (element i -> i +/- amount)."""
    for bit in range(DEG):
        lane = km.array([r[bit] for r in regs])
        if left:
            b.left_rotate(lane, amount)
        else:
            b.right_rotate(lane, amount)


def _rel(regs, k):
    """Static relabel: element i moves to i + k (mod len)."""
    k %= len(regs)
    return regs[-k:] + regs[:-k] if k else list(regs)


def cleanup(b, F, G, w, inverse: bool):
    if not inverse:
        _rotate(b, G, w, left=False)
        _rotate(b, G, w, left=False)
        G = _rel(G, 36)
        _masked_swap(b, F, G, w)
        _rotate(b, G, w, left=True)
        G = _rel(G, -1)
        sub = _rel(F[37:], -35)
        _rotate(b, sub, w, left=True)
        F = F[:37] + sub
    else:
        sub = F[37:]
        _rotate(b, sub, w, left=False)
        F = F[:37] + _rel(sub, 35)
        G = _rel(G, 1)
        _rotate(b, G, w, left=False)
        _masked_swap(b, F, G, w)
        G = _rel(G, -36)
        _rotate(b, G, w, left=True)
        _rotate(b, G, w, left=True)
    return F, G


def _masked_swap(b, F, G, w):
    for p in range(37, 71):
        mk = b.alloc_qubits(1)[0]
        b.flip_if_less_than(w, 71 - p, target=mk)        # w <= 70 - p
        b.cswap(mk, F[p], G[p])
        b.flip_if_less_than(w, 71 - p, target=mk)
        b.free(km.array([mk]))


def chien(b, F, G, E, field: km.GF2Field):
    sig = [F[K - k] for k in range(36)]       # sigma~_k
    om = [G[p] for p in range(35)]            # Omega~_p
    alpha = field.primitive_element
    odd = [sig[k] for k in range(3, 36, 2)]
    even = [sig[k] for k in range(2, 36, 2)]
    fwd_src = [q for r in odd for q in r] + [q for r in even for q in r] + [q for q in sig[1]] + [q for r in om[1:] for q in r]
    fwd_dst = [q for _ in odd for q in sig[1]] + [q for _ in even for q in sig[0]] + [q for q in sig[0]] + [q for _ in om[1:] for q in om[0]]
    fwd_src, fwd_dst = km.array(fwd_src), km.array(fwd_dst)
    rev_src, rev_dst = km.array(list(fwd_src)[::-1]), km.array(list(fwd_dst)[::-1])
    is_err = b.alloc_bits(1)
    z = b.alloc_qubits(1)[0]
    m = len(E)
    step = b.mark('forney_step')
    for j in range(m):
        with step:
            b.bit_store(is_err[0], True, control=E[j])
            b.push_condition(is_err[0])
            b.cx(fwd_src, fwd_dst)
            lad = _init_is_zero(b, sig[0], z)
            inv, st = gfq.init_inverse(b, sig[1], gate=z)
            gfq.phase(b, _mul(), om[0], inv, E[j])
            gfq.del_inverse(b, st)
            _del_is_zero(b, sig[0], z, lad)
            b.cx(rev_src, rev_dst)
            b.pop_condition()
            b.bit_store(is_err, False)
            if j + 1 < m:
                for k in range(1, 36):
                    b.gf2_imul(field.pow(alpha, k), target=sig[k], field=field)
                for p in range(1, 35):
                    b.gf2_imul(field.pow(alpha, p), target=om[p], field=field)
    if m > 1:
        for k in range(1, 36):
            b.gf2_idiv(field.pow(field.pow(alpha, k), m - 1), target=sig[k], field=field)
        for p in range(1, 35):
            b.gf2_idiv(field.pow(field.pow(alpha, p), m - 1), target=om[p], field=field)
    b.free(km.array([z]))
    b.free(is_err)


def build(m: int = 4095, n: int = NSYN) -> km.Circuit:
    field = km.GF2Field(DEG)
    b = km.CircuitBuilder()
    S = [b.create_quantum_register(DEG, name=f'S{i}') for i in range(n)]
    E = [b.create_classical_register(DEG, name=f'E{j}') for j in range(m)]
    F = [b.alloc_qubits(DEG) for _ in range(N)]
    G = [S[n - 1 - i] for i in range(n)] + [b.alloc_qubits(DEG) for _ in range(N - n)]
    F0, G0 = list(F), list(G)
    w = b.alloc_qubits(WBITS)
    ones = km.array([F[0][0], F[K][0]])
    b.x(ones)
    with b.mark('forward_eea'):
        for k in range(PAIRS):
            F, G = pair_forward(b, F, G, w, k)
    with b.mark('cleanup'):
        F, G = cleanup(b, F, G, w, inverse=False)
    with b.mark('chien_forney'):
        chien(b, F, G, E, field)
    with b.mark('uncleanup'):
        F, G = cleanup(b, F, G, w, inverse=True)
    with b.mark('reverse_eea'):
        for k in reversed(range(PAIRS)):
            F, G = pair_reverse(b, F, G, w, k)
    assert [r[0] for r in F] == [r[0] for r in F0] and [r[0] for r in G] == [r[0] for r in G0]
    b.x(ones)
    b.free(w)
    for r in F:
        b.free(r)
    for r in G[n:]:
        b.free(r)
    return b.finish_circuit()


if __name__ == '__main__':
    import time
    t = time.time()
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 4095
    c = build(m=m)
    c.to_file(sys.argv[1], format='kmb')
    print(f'{sys.argv[1]}: toffoli {c.operation_statistics().num_toffoli_like:,} qubits {c.num_qubits} '
          f'ops {len(c):,} ({time.time() - t:.0f}s)')
