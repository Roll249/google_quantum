"""Kickmix circuits for bilinear maps over GF(2) (GF(2^12) arithmetic for the DQI decoder).

A bilinear map z = f(x, y) is given by product forms U (K x nx), V (K x ny) and output matrix
W (nz x K):  p_k = (U_k . x)(V_k . y),  z = W p.  Each product costs one Toffoli:

  * the linear forms are formed in place: CNOT the other support bits onto a pivot bit,
    use the pivot, undo the CNOTs (no ancilla);
  * the output column W[:, k] is reached by fanning a pivot target bit out before and after
    the Toffoli, so target ^= p_k * W[:, k] with no ancilla;
  * erasing z = f(x, y) (x, y still present) is free: HMR z, then apply the phase
    (-1)^{r . z} = prod_k (-1)^{(r . W[:, k]) p_k} with classically conditioned CZs.

When x and y are the same register (quadratic maps), the y-form is formed on a one-qubit
ancilla.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np
import kickmix as km


@dataclasses.dataclass
class Bilinear:
    U: np.ndarray
    V: np.ndarray
    W: np.ndarray
    L: np.ndarray | None = None   # linear part (quadratic maps only): z ^= L x

    @property
    def K(self) -> int:
        return len(self.U)

    def simplified(self) -> 'Bilinear':
        """Drop zero columns of W and products with a zero form."""
        keep = [k for k in range(self.K) if self.W[:, k].any() and self.U[k].any() and self.V[k].any()]
        return Bilinear(self.U[keep], self.V[keep], self.W[:, keep], self.L)


def _form(b: km.CircuitBuilder, reg, vec, avoid=None):
    """Make some bit of `reg` hold vec . reg; returns (wire, undo)."""
    sup = [i for i in range(len(vec)) if vec[i]]
    cand = [i for i in sup if avoid is None or reg[i] is not avoid]
    p = cand[0]
    others = [i for i in sup if i != p]
    for i in others:
        b.cx(reg[i], reg[p])

    def undo():
        for i in reversed(others):
            b.cx(reg[i], reg[p])
    return reg[p], undo


def _fan(b: km.CircuitBuilder, reg, col):
    sup = [i for i in range(len(col)) if col[i]]
    p = sup[0]
    for i in sup[1:]:
        b.cx(reg[p], reg[i])
    return reg[p], lambda: [b.cx(reg[p], reg[i]) for i in sup[1:]]


def _product_wires(b, alg: Bilinear, x, y, k, same: bool):
    """Yields wires (xw, yw) holding U_k.x and V_k.y; returns undo callable."""
    if not same:
        xw, ux = _form(b, x, alg.U[k])
        yw, uy = _form(b, y, alg.V[k])
        return xw, yw, lambda: (uy(), ux())
    # quadratic: y-form on an ancilla
    anc = b.alloc_qubits(1)[0]
    for i in np.flatnonzero(alg.V[k]):
        b.cx(x[int(i)], anc)
    xw, ux = _form(b, x, alg.U[k])

    def undo():
        ux()
        for i in np.flatnonzero(alg.V[k]):
            b.cx(x[int(i)], anc)
        b.free(anc)
    return xw, anc, undo


def apply_acc(b: km.CircuitBuilder, alg: Bilinear, x, y, z) -> None:
    """z ^= f(x, y).  Pass y=None for a quadratic map of x."""
    same = y is None
    if same:
        y = x
    for k in range(alg.K):
        if not alg.W[:, k].any():
            continue
        xw, yw, undo = _product_wires(b, alg, x, y, k, same)
        zw, unfan = _fan(b, z, alg.W[:, k])
        b.ccx(xw, yw, zw)
        unfan()
        undo()
    if alg.L is not None:
        for i, j in zip(*np.nonzero(alg.L)):
            b.cx(x[int(j)], z[int(i)])


def init(b: km.CircuitBuilder, alg: Bilinear, x, y=None):
    z = b.alloc_qubits(alg.W.shape[0])
    apply_acc(b, alg, x, y, z)
    return z


def phase(b: km.CircuitBuilder, alg: Bilinear, x, y, mask_bits) -> None:
    """Global phase (-1)^{mask . f(x, y)} for classical mask bits (one per output bit)."""
    same = y is None
    if same:
        y = x
    par = b.alloc_bits(1)[0]
    for k in range(alg.K):
        sup = np.flatnonzero(alg.W[:, k])
        if len(sup) == 0:
            continue
        b.bit_store(par, False)
        for i in sup:
            b.push_condition(mask_bits[int(i)])
            b.x(par)
            b.pop_condition()
        b.push_condition(par)
        xw, yw, undo = _product_wires(b, alg, x, y, k, same)
        b.cz(xw, yw)
        undo()
        b.pop_condition()
    if alg.L is not None:
        # (-1)^{mask . L x}: Z on x_j when the parity of mask over column j of L is odd
        for j in range(alg.L.shape[1]):
            sup = np.flatnonzero(alg.L[:, j])
            if len(sup) == 0:
                continue
            b.bit_store(par, False)
            for i in sup:
                b.push_condition(mask_bits[int(i)])
                b.x(par)
                b.pop_condition()
            b.push_condition(par)
            b.z(x[j])
            b.pop_condition()
    b.free(km.array([par]))


def delete(b: km.CircuitBuilder, alg: Bilinear, x, y, z) -> None:
    """Erase z = f(x, y) (x, y unchanged) by X-basis measurement and CZ fixups: 0 Toffoli."""
    r = b.alloc_bits(len(z))
    b.hmr(z, r)
    phase(b, alg, x, y, r)
    b.free(z)
    b.free(r)


_ALGS = None


def alg(name: str) -> Bilinear:
    global _ALGS
    if _ALGS is None:
        import gfalg
        _ALGS = {k: Bilinear(np.asarray(U) % 2, np.asarray(V) % 2, np.asarray(W) % 2,
                             None if L is None else np.asarray(L) % 2)
                 for k, (U, V, W, L, *_) in gfalg.algorithms().items()}
    return _ALGS[name]


def mul_acc(b, x, y, z):
    """z ^= x * y (GF(2^12), 42 Toffolis, no ancilla)."""
    apply_acc(b, alg('mul'), x, y, z)


def init_inverse(b, x, gate=None):
    """Returns (xinv, scaffold) with xinv = gate * x^{-1} (x^{-1} := 0 at x = 0).

    Tower GF(8) < GF(64) < GF(2^12):  N = x^65, M = N^9, M^-1 = M^6, N^-1 = N^8 M^-1,
    x^-1 = x^64 N^-1.  75 Toffolis (+3 when gated); `del_inverse` is free.
    """
    steps = []
    N = init(b, alg('norm'), x); steps.append(('norm', x, None, N))
    M = init(b, alg('norm8'), N); steps.append(('norm8', N, None, M))
    Mi = init(b, alg('inv8'), M); steps.append(('inv8', M, None, Mi))
    if gate is not None:
        G = init(b, alg('gate8'), km.array([gate]), Mi); steps.append(('gate8', km.array([gate]), Mi, G))
        Mi = G
    Ni = init(b, alg('ninv'), N, Mi); steps.append(('ninv', N, Mi, Ni))
    Xi = init(b, alg('xinv'), x, Ni); steps.append(('xinv', x, Ni, Xi))
    return Xi, steps


def del_inverse(b, steps) -> None:
    for name, u, v, z in reversed(steps):
        delete(b, alg(name), u, v, z)


def init_inverse_compact(b, x, gate=None):
    """Like init_inverse, but only the 12-qubit result stays allocated (75 Toffolis now,
    ~39 more in del_inverse_compact to rebuild the chain for the free erasure)."""
    xi, steps = init_inverse(b, x, gate)
    for name, u, v, z in reversed(steps[:-1]):
        delete(b, alg(name), u, v, z)
    return xi, (x, gate, xi)


def del_inverse_compact(b, info) -> None:
    x, gate, xi = info
    steps = []
    N = init(b, alg('norm'), x); steps.append(('norm', x, None, N))
    M = init(b, alg('norm8'), N); steps.append(('norm8', N, None, M))
    Mi = init(b, alg('inv8'), M); steps.append(('inv8', M, None, Mi))
    if gate is not None:
        G = init(b, alg('gate8'), km.array([gate]), Mi); steps.append(('gate8', km.array([gate]), Mi, G))
        Mi = G
    Ni = init(b, alg('ninv'), N, Mi); steps.append(('ninv', N, Mi, Ni))
    delete(b, alg('xinv'), x, Ni, xi)
    for name, u, v, z in reversed(steps):
        delete(b, alg(name), u, v, z)
