"""Benchmark II decoder v3: decoder2's static divstep EEA extended to quotients of degree <= 2.

Differences from decoder2:
  * N = 74 slots per array (the static split P(step) stays valid for degree-2 rounds);
  * a swap may also happen on the odd step of a pair (a "0S" pair: delay + swap), and the pair
    following such a pair ("nn") never swaps.  With bprev = [previous pair was 0S] = [bf_1 != 0]
    at the pair start:  s_e = [g_0 != 0] & ~bprev,  s_o = [g_0 != 0] & ~s_e & ~bprev;
  * the pair-end recovery handles the three pair types (read off bf_1, bf_2, bf_3):
        Sn / S0 / 00 (bf_1 = 0):        c_e = bg_0 / bf_2,  c_o = (bg_1 + c_e bf_3) / bf_2
        0S           (bf_1 != 0):       c_e = 0,            c_o = bg_0 / bf_1
        nn           (bf_1 = bf_2 = 0): with B_j = bf_{j+3}, c0 = bg_0 / B_0,
                                        c_e = (bg_1 + c0 B_1) / B_0,
                                        c_o = (bg_2 + c0 B_2 + c_e B_1) / B_0
    and s_e = [bf_1 = 0][bf_2 != 0], s_o = [bf_1 != 0], bprev = [bf_1 = 0][bf_2 = 0][bf_3 != 0];
  * w = deg sigma is counted as sum over pairs of (s_e + s_o + bprev).
"""
from __future__ import annotations

import sys

import kickmix as km

import gfq
from decoder import _init_is_zero, _del_is_zero
from decoder2 import _mul, _rotate, _rel, DEG, NSYN, PAIRS

N = 74
K = N - 1
WBITS = 6
P = [70, 70, 70, 70, 69, 69, 68, 68, 67, 67, 66, 66, 65, 65, 64, 64, 63, 63, 62, 62, 61, 61, 60, 60,
     59, 59, 58, 58, 57, 57, 56, 56, 55, 55, 54, 54, 53, 53, 52, 52, 51, 51, 50, 50, 49, 49, 48, 48,
     47, 47, 46, 46, 45, 45, 44, 44, 43, 43, 42, 42, 41, 41, 40, 40, 39, 39, 38, 37, 37, 36]


class PlainZero:
    """z = [reg == 0] as a single qubit (ladder freed; uncompute recomputes it)."""

    def __init__(self, b, reg):
        self.b, self.reg = b, reg
        self.z = b.alloc_qubits(1)[0]
        zf = ZeroFlag(b, reg)
        b.cx(zf.z, self.z)
        zf.free()

    def free(self):
        zf = ZeroFlag(self.b, self.reg)
        self.b.cx(zf.z, self.z)
        zf.free()
        self.b.free(km.array([self.z]))


class ZeroFlag:
    """z = [reg == 0] with its AND ladder kept for free cleanup."""

    def __init__(self, b, reg):
        self.b, self.reg = b, reg
        self.z = b.alloc_qubits(1)[0]
        self.lad = _init_is_zero(b, reg, self.z)
        b.x(reg)                                   # restore reg while the flag is alive

    def free(self):
        self.b.x(self.reg)
        _del_is_zero(self.b, self.reg, self.z, self.lad)
        self.b.free(km.array([self.z]))


def _swap_flag(b, g0, blockers, s):
    """s ^= [g0 != 0] & ~blockers[0] & ~blockers[1] ... (s must be clean)."""
    zf = ZeroFlag(b, g0)
    ctl = [zf.z] + list(blockers)
    b.x(km.array(ctl))                       # all must be 1 = (g0 == 0) false, blockers false
    _and_all(b, ctl, s)
    b.x(km.array(ctl))
    zf.free()


def _and_all(b, ctl, target):
    """target ^= AND(ctl) for up to 3 controls, temporaries cleaned for free."""
    if len(ctl) == 1:
        b.cx(ctl[0], target)
    elif len(ctl) == 2:
        b.ccx(ctl[0], ctl[1], target)
    else:
        t = b.alloc_qubits(1)[0]
        b.init_and(ctl[0], ctl[1], t)
        b.ccx(t, ctl[2], target)
        b.del_and(ctl[0], ctl[1], t)
        b.free(km.array([t]))


def _cswap_arrays(b, s, F, G):
    for i in range(N):
        b.cswap(s, F[i], G[i])


def _updates(b, c, F, G, step):
    p = P[step]
    for i in range(N):
        if i < p:
            gfq.mul_acc(b, c, F[i], G[i])
        else:
            gfq.mul_acc(b, c, G[i], F[i])


def _step_forward(b, F, G, s, step):
    _cswap_arrays(b, s, F, G)
    inv0, st0 = gfq.init_inverse(b, F[0])
    c = gfq.init(b, _mul(), G[0], inv0)
    gfq.del_inverse(b, st0)
    _updates(b, c, F, G, step)
    G = G[1:] + [G[0]]
    return G, c


def _step_reverse(b, F, G, s, c, step):
    G = [G[-1]] + G[:-1]
    _updates(b, c, F, G, step)
    inv0, st0 = gfq.init_inverse(b, F[0])
    gfq.mul_acc(b, G[0], inv0, c)
    gfq.del_inverse(b, st0)
    b.free(c)
    _cswap_arrays(b, s, F, G)
    return G


def _recover(b, F, G, ce, co, z1, z2, compute: bool):
    """XOR the pair's quotient coefficients (from Bezout low coefficients) into ce, co.

    compute=True fills clean ce, co; compute=False erases them (same XOR blocks, other order:
    each pair type's c_o formula reads c_e)."""
    bg = lambda j: F[K - j]
    bf = lambda j: G[K - j]
    M = _mul()

    def order(first, second):
        (first(), second()) if compute else (second(), first())

    # nn pair: gate = z1 & z2, B_j = bf_{j+3}
    gC = b.alloc_qubits(1)[0]
    b.init_and(z1, z2, gC)
    iv, st = gfq.init_inverse_compact(b, bf(3), gate=gC)
    c0 = gfq.init(b, M, bg(0), iv)

    def nn_ce():
        t = gfq.init(b, M, c0, bf(4))
        b.cx(bg(1), t)
        gfq.mul_acc(b, t, iv, ce)
        b.cx(bg(1), t)
        gfq.delete(b, M, c0, bf(4), t)

    def nn_co():
        v = gfq.init(b, M, c0, bf(5))
        gfq.mul_acc(b, ce, bf(4), v)
        b.cx(bg(2), v)
        gfq.mul_acc(b, v, iv, co)
        b.cx(bg(2), v)
        gfq.mul_acc(b, ce, bf(4), v)
        gfq.delete(b, M, c0, bf(5), v)

    order(nn_ce, nn_co)
    gfq.delete(b, M, bg(0), iv, c0)
    gfq.del_inverse_compact(b, st)
    b.del_and(z1, z2, gC)
    b.free(km.array([gC]))
    # Sn / S0 / 00: gate = z1
    iv, st = gfq.init_inverse_compact(b, bf(2), gate=z1)

    def a_ce():
        gfq.mul_acc(b, bg(0), iv, ce)

    def a_co():
        t = gfq.init(b, M, ce, bf(3))
        b.cx(bg(1), t)
        gfq.mul_acc(b, t, iv, co)
        b.cx(bg(1), t)
        gfq.delete(b, M, ce, bf(3), t)

    order(a_ce, a_co)
    gfq.del_inverse_compact(b, st)
    # 0S: c_o = bg_0 / bf_1 (the inverse is zero unless bf_1 != 0)
    iv, st = gfq.init_inverse_compact(b, bf(1))
    gfq.mul_acc(b, bg(0), iv, co)
    gfq.del_inverse_compact(b, st)


def _end_flags(b, G, s_e, s_o, bprev, z1, z2):
    """XOR the state-derived pair flags into s_e, s_o, bprev."""
    bf = lambda j: G[K - j]
    # s_o ^= ~z1
    b.cx(z1, s_o)
    b.x(s_o)
    # s_e ^= z1 & ~z2
    b.x(z2)
    b.ccx(z1, z2, s_e)
    b.x(z2)
    # bprev ^= z1 & z2 & ~z3
    z3 = ZeroFlag(b, bf(3))
    b.x(z3.z)
    _and_all(b, [z1, z2, z3.z], bprev)
    b.x(z3.z)
    z3.free()


def _count(b, w, flags, sign):
    for f in flags:
        if sign > 0:
            b.iadd(1, target=w, control=f)
        else:
            b.isub(1, target=w, control=f)


def pair_forward(b, F, G, w, k):
    bf = lambda j: G[K - j]
    s_e, s_o, bprev = (b.alloc_qubits(1)[0] for _ in range(3))
    zb = ZeroFlag(b, bf(1))
    b.cx(zb.z, bprev)
    b.x(bprev)                                 # bprev = [bf_1 != 0]
    zb.free()
    _swap_flag(b, G[0], [bprev], s_e)
    G, ce = _step_forward(b, F, G, s_e, 2 * k)
    _swap_flag(b, G[0], [s_e, bprev], s_o)
    G, co = _step_forward(b, F, G, s_o, 2 * k + 1)
    _count(b, w, (s_e, s_o, bprev), +1)
    zf1, zf2 = PlainZero(b, G[K - 1]), PlainZero(b, G[K - 2])
    _recover(b, F, G, ce, co, zf1.z, zf2.z, compute=False)
    b.free(ce + co)
    _end_flags(b, G, s_e, s_o, bprev, zf1.z, zf2.z)
    zf2.free(); zf1.free()
    b.free(km.array([s_e, s_o, bprev]))
    return F, G


def pair_reverse(b, F, G, w, k):
    s_e, s_o, bprev = (b.alloc_qubits(1)[0] for _ in range(3))
    zf1, zf2 = PlainZero(b, G[K - 1]), PlainZero(b, G[K - 2])
    _end_flags(b, G, s_e, s_o, bprev, zf1.z, zf2.z)
    ce, co = b.alloc_qubits(DEG), b.alloc_qubits(DEG)
    _recover(b, F, G, ce, co, zf1.z, zf2.z, compute=True)
    zf2.free(); zf1.free()
    _count(b, w, (s_e, s_o, bprev), -1)
    G = _step_reverse(b, F, G, s_o, co, 2 * k + 1)
    _swap_flag(b, G[0], [s_e, bprev], s_o)
    G = _step_reverse(b, F, G, s_e, ce, 2 * k)
    _swap_flag(b, G[0], [bprev], s_e)
    bf = lambda j: G[K - j]
    zb = ZeroFlag(b, bf(1))
    b.x(bprev)
    b.cx(zb.z, bprev)
    zb.free()
    b.free(km.array([s_e, s_o, bprev]))
    return F, G


def _masked_swap(b, F, G, w):
    for p in range(K - 35, 71):
        mk = b.alloc_qubits(1)[0]
        b.flip_if_less_than(w, 71 - p, target=mk)        # w <= 70 - p
        b.cswap(mk, F[p], G[p])
        b.flip_if_less_than(w, 71 - p, target=mk)
        b.free(km.array([mk]))


def cleanup(b, F, G, w, inverse: bool):
    if not inverse:
        _rotate(b, G, w, left=False)
        _rotate(b, G, w, left=False)
        G = _rel(G, 36)
        _masked_swap(b, F, G, w)
        _rotate(b, G, w, left=True)
        G = _rel(G, -1)
        sub = _rel(F[K - 35:], -35)
        _rotate(b, sub, w, left=True)
        F = F[:K - 35] + sub
    else:
        sub = F[K - 35:]
        _rotate(b, sub, w, left=False)
        F = F[:K - 35] + _rel(sub, 35)
        G = _rel(G, 1)
        _rotate(b, G, w, left=False)
        _masked_swap(b, F, G, w)
        G = _rel(G, -36)
        _rotate(b, G, w, left=True)
        _rotate(b, G, w, left=True)
    return F, G


def chien(b, F, G, E, field):
    import decoder2
    decoder2.K = K
    decoder2.chien(b, F, G, E, field)


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
    c = build()
    c.to_file(sys.argv[1], format='kmb')
    print(f'{sys.argv[1]}: toffoli {c.operation_statistics().num_toffoli_like:,} qubits {c.num_qubits} '
          f'ops {len(c):,} ({time.time() - t:.0f}s)')
