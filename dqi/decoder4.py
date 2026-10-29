"""Benchmark II decoder v4: rescaled reversible Berlekamp-Massey (after Gu et al. 2026, SI I.1).

Key equation by 70 BM rounds on sigma, tau (36 coefficient registers each, tau cyclic):

    tau <- z tau                       (free relabel)
    Delta = sum_i sigma_i S_{r-i}      (i <= k_{r-1}, adaptive range)
    replace = Delta != 0 and d >= 0,  retain = Delta != 0 and d < 0
    p = Delta sigma_0 (= alpha, since sigma_0 delta = 1),  m = replace ? p^-1 : p
    replace: swap sigma <-> tau;  then one shared update sigma ^= m tau  (i <= k_r)
    d <- replace ? -d : d + 1

Reversibility: replace <=> (tau_0 != 0 and d <= 0) after the round, and then Delta = 1 / sigma_0;
a retain round pushes Delta (with its round index) on a one-entry stack.  Shots needing a
second stack entry (~1e-4) fail.  Then sigma is normalised (sigma_j <- sigma_j / sigma_0,
sigma_0 kept), omega = sigma S mod z^35 overwrites S_1..S_35 in place, and the Chien/Forney
scan of decoder2 kicks back the phase; everything is uncomputed in reverse.
"""
from __future__ import annotations

import sys

import kickmix as km

import gfq
from decoder3 import PlainZero
from decoder2 import _mul

DEG = 12
NSYN = 70
L = 36
DBITS = 8
IBITS = 7


def k_upd(r: int) -> int:
    return min(r // 2 + 1, L - 1)


def k_dis(r: int) -> int:
    return 0 if r == 1 else k_upd(r - 1)


class State:
    def __init__(self, b, S):
        self.b, self.S = b, S
        self.sig = [b.alloc_qubits(DEG) for _ in range(L)]
        self.tau = [b.alloc_qubits(DEG) for _ in range(L)]
        self.d = b.alloc_qubits(DBITS)
        self.hst = b.alloc_qubits(DEG)
        self.hidx = b.alloc_qubits(IBITS)
        self.ones = km.array([self.sig[0][0], self.tau[0][0]])
        b.x(self.ones)

    def free(self):
        b = self.b
        b.x(self.ones)
        for r in self.sig + self.tau:
            b.free(r)
        b.free(self.d); b.free(self.hst); b.free(self.hidx)


def _discrepancy(b, st, r, D):
    for i in range(min(r - 1, k_dis(r)) + 1):
        gfq.mul_acc(b, st.sig[i], st.S[r - 1 - i], D)


def _flags_from_delta(b, st, D, rep, ret, compute: bool):
    """rep ^= [D != 0][d >= 0], ret ^= [D != 0][d < 0] (init_and / del_and)."""
    sign = st.d[DBITS - 1]
    zf = PlainZero(b, D)
    b.x(zf.z)
    b.x(sign)
    (b.init_and if compute else b.del_and)(zf.z, sign, rep)
    b.x(sign)
    (b.init_and if compute else b.del_and)(zf.z, sign, ret)
    b.x(zf.z)
    zf.free()


def _flags_from_state(b, st, r, rep, ret):
    """rep ^= [tau_0 != 0][d <= 0], ret ^= [hidx == r]."""
    sign = st.d[DBITS - 1]
    zt = PlainZero(b, st.tau[0])
    zd = PlainZero(b, st.d)
    t = b.alloc_qubits(1)[0]
    b.x(km.array([sign, zd.z]))
    b.init_and(sign, zd.z, t)                 # t = d > 0
    b.x(km.array([zt.z, t]))
    b.ccx(zt.z, t, rep)
    b.x(km.array([zt.z, t]))
    b.del_and(sign, zd.z, t)
    b.x(km.array([sign, zd.z]))
    b.free(km.array([t]))
    zd.free()
    zt.free()
    b.flip_if_equal(st.hidx, r, target=ret)


def _push(b, st, r, ret, D):
    b.cswap(ret, D, st.hst)
    for bit in range(IBITS):
        if (r >> bit) & 1:
            b.cx(ret, st.hidx[bit])


def _d_update(b, st, rep, inverse: bool):
    if not inverse:
        b.cx(km.array([rep] * DBITS), st.d)
        b.iadd(1, target=st.d)
    else:
        b.isub(1, target=st.d)
        b.cx(km.array([rep] * DBITS), st.d)


def _core(b, st, r, D, rep, forward: bool):
    """With D = Delta and the flags present: apply (forward) or undo the shared update."""
    M = _mul()
    sig, tau = st.sig, st.tau
    k = k_upd(r)
    if forward:
        p = gfq.init(b, M, D, sig[0])
    else:
        b.cswap(rep, sig[0], tau[0])
        p = gfq.init(b, M, D, sig[0])
        b.cswap(rep, sig[0], tau[0])
    pinv, pst = gfq.init_inverse_compact(b, p)
    b.cswap(rep, p, pinv)
    if forward:
        for i in range(k + 1):
            b.cswap(rep, sig[i], tau[i])
        for i in range(k + 1):
            gfq.mul_acc(b, p, tau[i], sig[i])
    else:
        for i in range(k + 1):
            gfq.mul_acc(b, p, tau[i], sig[i])
        for i in range(k + 1):
            b.cswap(rep, sig[i], tau[i])
    b.cswap(rep, p, pinv)
    gfq.del_inverse_compact(b, pst)
    if forward:
        b.cswap(rep, sig[0], tau[0])
        gfq.delete(b, M, D, sig[0], p)
        b.cswap(rep, sig[0], tau[0])
    else:
        gfq.delete(b, M, D, sig[0], p)


def _replace_delta(b, st, rep, D):
    """D ^= rep / sigma_0."""
    iv, ist = gfq.init_inverse_compact(b, st.sig[0], gate=rep)
    b.cx(iv, D)
    gfq.del_inverse_compact(b, ist)


def round_forward(b, st, r):
    st.tau = [st.tau[-1]] + st.tau[:-1]
    D = b.alloc_qubits(DEG)
    _discrepancy(b, st, r, D)
    rep, ret = b.alloc_qubits(1)[0], b.alloc_qubits(1)[0]
    _flags_from_delta(b, st, D, rep, ret, compute=True)
    _core(b, st, r, D, rep, forward=True)
    _replace_delta(b, st, rep, D)
    _push(b, st, r, ret, D)
    b.free(D)
    _d_update(b, st, rep, inverse=False)
    _flags_from_state(b, st, r, rep, ret)
    b.free(km.array([rep, ret]))


def round_reverse(b, st, r):
    rep, ret = b.alloc_qubits(1)[0], b.alloc_qubits(1)[0]
    _flags_from_state(b, st, r, rep, ret)
    _d_update(b, st, rep, inverse=True)
    D = b.alloc_qubits(DEG)
    _push(b, st, r, ret, D)
    _replace_delta(b, st, rep, D)
    _core(b, st, r, D, rep, forward=False)
    _flags_from_delta(b, st, D, rep, ret, compute=False)
    b.free(km.array([rep, ret]))
    _discrepancy(b, st, r, D)
    b.free(D)
    st.tau = st.tau[1:] + [st.tau[0]]


def normalize(b, st, inverse: bool):
    """sigma_j <- sigma_j / sigma_0 for j >= 1 (inverse: back)."""
    M = _mul()
    c, cst = gfq.init_inverse_compact(b, st.sig[0])
    for j in range(1, L):
        old = st.sig[j]
        if not inverse:
            t = gfq.init(b, M, old, c)
            gfq.mul_acc(b, t, st.sig[0], old)
        else:
            t = gfq.init(b, M, old, st.sig[0])
            gfq.mul_acc(b, t, c, old)
        b.free(old)
        st.sig[j] = t
    gfq.del_inverse_compact(b, cst)


def omega(b, st, inverse: bool):
    """S_{i+1} <- omega_i = S_{i+1} + sum_{j=1}^{i} sigma'_j S_{i+1-j}, i < 35."""
    order = range(35) if inverse else range(34, -1, -1)
    for i in order:
        for j in range(1, min(i, L - 1) + 1):
            gfq.mul_acc(b, st.sig[j], st.S[i - j], st.S[i])


def scan(b, st, E, field):
    sig, S = st.sig, st.S
    om = [S[i] for i in range(35)]
    g = field.primitive_element
    ginv = field.div(1, g)
    odd = [sig[k] for k in range(3, L, 2)]
    even = [sig[k] for k in range(4, L, 2)]
    src = [q for r in odd for q in r] + [q for r in even for q in r] + list(sig[1]) + [q for r in om[1:] for q in r]
    dst = [q for _ in odd for q in sig[1]] + [q for _ in even for q in sig[2]] + list(sig[2]) + [q for _ in om[1:] for q in om[0]]
    fwd_src, fwd_dst = km.array(src), km.array(dst)
    rev_src, rev_dst = km.array(src[::-1]), km.array(dst[::-1])
    is_err = b.alloc_bits(1)
    z = b.alloc_qubits(1)[0]
    m = len(E)
    mark = b.mark('forney_step')
    from decoder import _init_is_zero, _del_is_zero
    for j in range(m):
        with mark:
            b.bit_store(is_err[0], True, control=E[j])
            b.push_condition(is_err[0])
            b.cx(fwd_src, fwd_dst)
            b.x(sig[2][0])                          # + 1 (sigma'_0)
            lad = _init_is_zero(b, sig[2], z)
            inv, ist = gfq.init_inverse(b, sig[1], gate=z)
            gfq.phase(b, _mul(), om[0], inv, E[j])
            gfq.del_inverse(b, ist)
            _del_is_zero(b, sig[2], z, lad)
            b.x(sig[2][0])
            b.cx(rev_src, rev_dst)
            b.pop_condition()
            b.bit_store(is_err, False)
            if j + 1 < m:
                for k in range(1, L):
                    b.gf2_imul(field.pow(ginv, k), target=sig[k], field=field)
                for i in range(35):
                    b.gf2_imul(field.pow(ginv, i + 1), target=om[i], field=field)
    if m > 1:
        for k in range(1, L):
            b.gf2_idiv(field.pow(field.pow(ginv, k), m - 1), target=sig[k], field=field)
        for i in range(35):
            b.gf2_idiv(field.pow(field.pow(ginv, i + 1), m - 1), target=om[i], field=field)
    b.free(km.array([z]))
    b.free(is_err)


def build(m: int = 4095, n: int = NSYN) -> km.Circuit:
    field = km.GF2Field(DEG)
    b = km.CircuitBuilder()
    S = [b.create_quantum_register(DEG, name=f'S{i}') for i in range(n)]
    E = [b.create_classical_register(DEG, name=f'E{j}') for j in range(m)]
    st = State(b, S)
    tau0 = list(st.tau)
    with b.mark('bm_forward'):
        for r in range(1, 71):
            round_forward(b, st, r)
    with b.mark('normalize'):
        normalize(b, st, inverse=False)
    with b.mark('omega'):
        omega(b, st, inverse=False)
    with b.mark('chien_forney'):
        scan(b, st, E, field)
    with b.mark('unomega'):
        omega(b, st, inverse=True)
    with b.mark('unnormalize'):
        normalize(b, st, inverse=True)
    with b.mark('bm_reverse'):
        for r in range(70, 0, -1):
            round_reverse(b, st, r)
    assert [x[0] for x in st.tau] == [x[0] for x in tau0]
    st.free()
    return b.finish_circuit()


if __name__ == '__main__':
    import time
    t = time.time()
    c = build()
    c.to_file(sys.argv[1], format='kmb')
    print(f'{sys.argv[1]}: toffoli {c.operation_statistics().num_toffoli_like:,} qubits {c.num_qubits} '
          f'ops {len(c):,} ({time.time() - t:.0f}s)')
