import sys; sys.path.insert(0, 'dqi'); sys.path.insert(0, 'data/verifiers/dqi')
import numpy as np, kickmix as km
import decoder3 as D3
import gfalg as G
from divstep_model import mul, inv, n
sys.argv = [sys.argv[0], '0', '0', '74']
import general_model as GM

npairs = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != '0' else 1
def build(npairs):
    b = km.CircuitBuilder()
    S = [b.create_quantum_register(12, name=f'S{i}') for i in range(70)]
    F = [b.alloc_qubits(12) for _ in range(D3.N)]
    Gl = [S[69 - i] for i in range(70)] + [b.alloc_qubits(12) for _ in range(D3.N - 70)]
    w = b.alloc_qubits(D3.WBITS)
    b.x(km.array([F[0][0], F[D3.K][0]]))
    for k in range(npairs):
        F, Gl = D3.pair_forward(b, F, Gl, w, k)
    return b.finish_circuit(), S, F, Gl, w

rng = np.random.default_rng(3)
shots = []
for t in range(64):
    wt = 33
    pos = rng.choice(4095, wt, replace=False)
    err = {int(p): int(rng.integers(1, 4096)) for p in pos}
    Sv = [0] * 70
    for j, v in err.items():
        for i in range(70):
            Sv[i] ^= mul(v, G.gpow(2, ((i + 1) * j) % 4095))
    shots.append(Sv)
for npairs in (1, 2, 3):
    c, S, F, Gl, w = build(npairs)
    sim = km.Simulator(batch_size=64, seed=1); sim.use_same_registers_as(c)
    Sb = np.array(shots)
    for i in range(70):
        sim.write_across_shots(S[i], ((Sb[:, i][None, :] >> np.arange(12)[:, None]) & 1) == 1)
    sim.do(c)
    rd = lambda r: (sim.read_across_shots(r).astype(np.int64) * (1 << np.arange(12))[:, None]).sum(0)
    Fq = np.array([rd(r) for r in F]); Gq = np.array([rd(r) for r in Gl])
    # model
    bad = 0
    for t in range(64):
        Fm = [0] * D3.N; Gm = [0] * D3.N; Fm[0] = 1; Fm[D3.K] = 1
        for i in range(70): Gm[i] = shots[t][69 - i]
        delta = 1
        for step in range(2 * npairs):
            s = delta > 0 and Gm[0] != 0
            if s: Fm, Gm = Gm, Fm; delta = 1 - delta
            else: delta += 1
            cc = mul(Gm[0], inv(Fm[0]))
            for i in range(D3.N):
                if i < D3.P[step]: Gm[i] ^= mul(cc, Fm[i])
                else: Fm[i] ^= mul(cc, Gm[i])
            Gm = Gm[1:] + [0]
        if list(Fq[:, t]) != Fm or list(Gq[:, t]) != Gm:
            bad += 1
            if bad == 1:
                print('  F diff idx', [i for i in range(D3.N) if Fq[i, t] != Fm[i]][:8], 'G diff', [i for i in range(D3.N) if Gq[i, t] != Gm[i]][:8])
    anc = [km.q(k) for k in range(c.num_qubits)]
    used = set(q for r in F + Gl for q in r) | set(w)
    junk = sim.read_across_shots(km.array([q for q in anc if q not in used and q not in set(x for r in S for x in r)]))
    print('pairs', npairs, 'mismatching shots', bad, 'dirty ancilla shots', int(junk.any(axis=0).sum()), 'w', rd(w)[:5] if False else (sim.read_across_shots(w).astype(int) * (1 << np.arange(D3.WBITS))[:, None]).sum(0)[:5])
