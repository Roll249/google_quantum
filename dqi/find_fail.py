import sys; sys.path.insert(0, 'dqi'); sys.path.insert(0, 'data/verifiers/dqi')
import numpy as np, kickmix as km
import verify_submission_dqi as V
import gfalg as G
from divstep_model import mul, inv
c = km.Circuit.from_file(sys.argv[1])
spec = V.ProblemSpecDQI.benchmark_instance()
regs = c.register_data
S = [list(regs[f'S{i}']) for i in range(70)]
E = [list(regs[f'E{j}']) for j in range(4095)]
wp = V.optimal_dqi_weights(4095, 35, 12, 2016) ** 2
rng = np.random.default_rng(int(sys.argv[2]))
B = 256
fails = []
for rep in range(int(sys.argv[3])):
    errs = []
    syn = np.zeros((B, 70), dtype=np.int64); mb = rng.integers(0, 4096, (B, 4095)); ph = np.zeros(B, bool)
    for t in range(B):
        w = int(rng.choice(36, p=wp)); pos = rng.choice(4095, w, replace=False)
        err = {int(p): int(rng.integers(1, 4096)) for p in pos}
        for j, v in err.items():
            for i in range(70): syn[t, i] ^= mul(v, G.gpow(2, ((i + 1) * j) % 4095))
        ph[t] = sum(bin(v & int(mb[t, j])).count('1') for j, v in err.items()) & 1
        errs.append(err)
    sim = km.Simulator(batch_size=B, seed=rep); sim.use_same_registers_as(c)
    sim.write_across_shots(km.array([q for r in S for q in r]), V.to_bits(syn, 12))
    sim.write_across_shots(km.array([q for r in E for q in r]), V.to_bits(mb, 12))
    sim.do(c)
    bad = np.flatnonzero(sim.read_phase_flipped_across_shots() ^ ph)
    for t in bad:
        fails.append((errs[t], syn[t].tolist()))
print('fails', len(fails))
import pickle; pickle.dump(fails, open('runs/v3_fails.pkl', 'wb'))
for err, Sv in fails[:6]:
    f = [1]; g = Sv[::-1]; d = 1; pat = ''
    for step in range(70):
        sw = d > 0 and g[0] != 0
        if sw: f, g, d = g, f, -d
        cc = mul(g[0], inv(f[0])); pat += 'S' if sw else ('n' if cc else '0')
        L = max(len(f), len(g)); f += [0] * (L - len(f)); g += [0] * (L - len(g))
        g = [g[i] ^ mul(cc, f[i]) for i in range(L)][1:]
        d += 1
    print(len(err), pat)
