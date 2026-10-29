import sys; sys.path.insert(0, 'dqi')
import numpy as np, kickmix as km
import decoder3 as D3
from decoder3 import *
import gfalg as G
from divstep_model import mul, inv

def pf(b, F, Gl, w, k):
    bf = lambda j: Gl[K - j]
    s_e, s_o, bprev = (b.alloc_qubits(1)[0] for _ in range(3))
    zb = ZeroFlag(b, bf(1)); b.cx(zb.z, bprev); b.x(bprev); zb.free()
    D3._swap_flag(b, Gl[0], [bprev], s_e)
    Gl, ce = D3._step_forward(b, F, Gl, s_e, 2 * k)
    D3._swap_flag(b, Gl[0], [s_e, bprev], s_o)
    Gl, co = D3._step_forward(b, F, Gl, s_o, 2 * k + 1)
    snap = (s_e, s_o, bprev)
    zf1, zf2 = ZeroFlag(b, Gl[K - 1]), ZeroFlag(b, Gl[K - 2])
    ce2 = b.alloc_qubits(12); co2 = b.alloc_qubits(12)
    b.cx(ce, ce2); b.cx(co, co2)
    D3._recover(b, F, Gl, ce, co, zf1.z, zf2.z, compute=False)
    D3._end_flags(b, Gl, s_e, s_o, bprev, zf1.z, zf2.z)
    zf2.free(); zf1.free()
    return F, Gl, ce, co, ce2, co2, s_e, s_o, bprev

b = km.CircuitBuilder()
S = [b.create_quantum_register(12, name=f'S{i}') for i in range(70)]
F = [b.alloc_qubits(12) for _ in range(N)]
Gl = [S[69 - i] for i in range(70)] + [b.alloc_qubits(12) for _ in range(N - 70)]
w = b.alloc_qubits(WBITS)
b.x(km.array([F[0][0], F[K][0]]))
F, Gl, ce, co, ce2, co2, s_e, s_o, bprev = pf(b, F, Gl, w, 0)
c = b.finish_circuit()
rng = np.random.default_rng(3)
shots = []
for t in range(16):
    pos = rng.choice(4095, 33, replace=False); err = {int(p): int(rng.integers(1, 4096)) for p in pos}
    Sv = [0] * 70
    for j, v in err.items():
        for i in range(70): Sv[i] ^= mul(v, G.gpow(2, ((i + 1) * j) % 4095))
    shots.append(Sv)
sim = km.Simulator(batch_size=16, seed=1); sim.use_same_registers_as(c)
Sb = np.array(shots)
for i in range(70): sim.write_across_shots(S[i], ((Sb[:, i][None, :] >> np.arange(12)[:, None]) & 1) == 1)
sim.do(c)
rd = lambda r: (sim.read_across_shots(r).astype(np.int64) * (1 << np.arange(len(r)))[:, None]).sum(0)
print('ce after', rd(ce)[:4], 'co after', rd(co)[:4])
print('ce orig', rd(ce2)[:4], 'co orig', rd(co2)[:4])
print('flags', rd(km.array([s_e]))[:4], rd(km.array([s_o]))[:4], rd(km.array([bprev]))[:4])
bfv = lambda j: rd(Gl[K - j]); bgv = lambda j: rd(F[K - j])
for t in range(2):
    b2, b3, g0, g1 = int(bfv(2)[t]), int(bfv(3)[t]), int(bgv(0)[t]), int(bgv(1)[t])
    iv = inv(b2); c1 = mul(g0, iv); c2 = mul(g1 ^ mul(c1, b3), iv)
    print('model recover', c1, c2, 'bf1', bfv(1)[t])
