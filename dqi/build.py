"""Build the improved DQI decoder circuit:  python dqi/build.py out.kmb"""
import sys, time
sys.path.insert(0, 'dqi')
import kickmix as km
import decoder as D

t = time.time()
m, n, deg = 4095, 70, 12
b = km.CircuitBuilder()
S = [b.create_quantum_register(deg, name=f'S{i}') for i in range(n)]
E = [b.create_classical_register(deg, name=f'E{j}') for j in range(m)]
D.ProosAndZalkaDecoder(n, km.GF2Field(deg)).build_syndrome_decoding_mbuc(b, syndromes=S, error_mbuc=E, n_iters=D.default_n_iters(n))
c = b.finish_circuit()
c.to_file(sys.argv[1], format='kmb')
print(f'{sys.argv[1]}: toffoli {c.operation_statistics().num_toffoli_like:,} qubits {c.num_qubits} ops {len(c):,} ({time.time()-t:.0f}s)')
