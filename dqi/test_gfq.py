import sys; sys.path.insert(0, 'dqi')
import numpy as np, kickmix as km
import gfq, gfalg as G

def run(build, n_in, shots_vals):
    b = km.CircuitBuilder()
    regs = [b.create_quantum_register(12, name=f'r{i}') for i in range(n_in)]
    out = build(b, regs)
    c = b.finish_circuit()
    st = c.operation_statistics()
    N = shots_vals[0].shape[0]
    sim = km.Simulator(batch_size=N, seed=3)
    sim.use_same_registers_as(c)
    for r, v in zip(regs, shots_vals):
        sim.write_across_shots(r, ((v[None, :] >> np.arange(12)[:, None]) & 1) == 1)
    sim.do(c)
    res = [ (sim.read_across_shots(r).astype(np.int64) * (1 << np.arange(12))[:, None]).sum(0) for r in regs]
    return st.num_toffoli_like, c.num_qubits, res, sim

# 1. mul_acc: r2 ^= r0 * r1
rng = np.random.default_rng(0)
x = rng.integers(0, 4096, 4096); y = rng.integers(0, 4096, 4096); z = rng.integers(0, 4096, 4096)
tof, q, res, _ = run(lambda b, r: gfq.mul_acc(b, r[0], r[1], r[2]), 3, [x, y, z])
print('mul_acc toffoli', tof, 'qubits', q, 'ok', (res[2] == z ^ G.gmul_vec(x, y)).all() and (res[0] == x).all() and (res[1] == y).all())

# 2. inverse into r1 (r1 ^= x^-1 via init/copy/del)
xs = np.arange(4096)
def inv_build(b, r):
    xi, steps = gfq.init_inverse(b, r[0])
    b.cx(xi, r[1])
    gfq.del_inverse(b, steps)
tof, q, res, sim = run(inv_build, 2, [xs, np.zeros(4096, dtype=np.int64)])
exp = np.array([G.ginv(int(v)) if v else 0 for v in xs])
junk = sim.read_across_shots(km.array([km.q(k) for k in range(24, q)])) if q > 24 else np.zeros((0, 4096), bool)
print('inverse toffoli', tof, 'qubits', q, 'value ok', (res[1] == exp).all(), 'clean', not junk.any(),
      'phase ok', not sim.read_phase_flipped_across_shots().any())
