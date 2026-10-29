"""How many synchronized-EEA iterations do test shots actually need? (reads halting_counter)."""
import sys, numpy as np, kickmix as km
sys.path.insert(0, 'dqi'); sys.path.insert(0, 'data/verifiers/dqi')
from ref_copy import ProosAndZalkaDecoder, default_n_iters
import verify_submission_dqi as V

n, deg, m = 70, 12, 4095
gf = km.GF2Field(deg)
NI = default_n_iters(n)
b = km.CircuitBuilder()
S = [b.create_quantum_register(deg, name=f'S{i}') for i in range(n)]
dec = ProosAndZalkaDecoder(n, gf)
# replicate build_syndrome_decoding_mbuc up to the forward EEA
mm = n + 3
reg_a = [b.alloc_qubits(deg) if i < 3 else S[i - 3] for i in range(mm)]
reg_b = [b.alloc_qubits(deg) for _ in range(mm)]
logm = mm.bit_length() + 1
n_A, n_B, n_a, n_b, n_q = [b.alloc_qubits(logm) for _ in range(5)]
flag = b.alloc_qubits(1)[0]; counter = b.alloc_qubits(2); hc = b.alloc_qubits(NI.bit_length())
b.x(km.array([reg_a[0][0], reg_b[mm - 1][0], n_a[0], flag]))
b.gf2_iadd(n, target=n_A); b.gf2_iadd(n + 1, target=n_B)
dec._initial_shift_loop(b, reg_a, n_A, is_adjoint=False)
dec._synchronized_eea(b, reg_a, reg_b, n_A, n_B, n_a, n_b, n_q, flag, counter, hc, n_iters=NI, is_adjoint=False)
c = b.finish_circuit()
spec = V.ProblemSpecDQI.benchmark_instance()
tables = V.FieldTables(gf)
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
B = 256
res = []
for rep in range(int(sys.argv[2]) if len(sys.argv) > 2 else 4):
    syn, mb, ph = V.sample_shots(spec, tables, rng, B)
    w = None
    sim = km.Simulator(batch_size=B, seed=rep)
    sim.use_same_registers_as(c)
    sim.write_across_shots(km.array([q for r in S for q in r]), V.to_bits(syn, deg))
    sim.do(c)
    bits = sim.read_across_shots(hc)  # (nbits, B)
    val = (bits.astype(np.int64) * (1 << np.arange(len(hc)))[:, None]).sum(0)
    res.extend((NI - val).tolist())
res = np.array(res)
print('needed iterations: max', res.max(), 'quantiles', np.percentile(res, [50, 90, 99, 99.9]))
print(np.bincount(res)[-40:])
