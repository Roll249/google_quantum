import sys, itertools, json, math, multiprocessing as mp
sys.path.insert(0, 'invsqrt')
from concurrent.futures import ProcessPoolExecutor
import gen_poly3 as G

def run(d):
    try:
        s, info = G.estimate(d)
    except Exception as ex:
        return (math.inf, str(d), repr(ex)[:80])
    return (s, str(d), info)

designs = []
for k, gb, use_and in itertools.product((6, 7, 8), (26, 27, 28), (True, False)):
    nd = 21 - k
    for t1 in range(gb - 9, gb - 4):
        for t2 in range(t1 - 11, t1 - 6):
            for g0, g1 in ((2, 1), (3, 2), (4, 2)):
                for d1 in range(max(3, t2 - k - 2), min(nd + 1, t2 - k + 4)):
                    designs.append(G.Design(k=k, degree=2, g_bits=gb, t_bits=(t1, t2), guard=(g0, g1),
                                            dbits=(nd, d1), use_and=use_and))
for k, gb, use_and in itertools.product((4, 5, 6), (26, 27, 28), (True, False)):
    nd = 21 - k
    for t1 in range(gb - 7, gb - 3):
        for t2 in range(t1 - 8, t1 - 3, 2):
            for t3 in range(t2 - 10, t2 - 4, 2):
                for d1 in (nd - 2, nd):
                    for d2 in range(max(3, t3 - k - 2), min(nd + 1, t3 - k + 3), 2):
                        designs.append(G.Design(k=k, degree=3, g_bits=gb, t_bits=(t1, t2, t3), guard=(3, 2, 2),
                                                dbits=(nd, d1, d2), use_and=use_and))
print(len(designs), flush=True)
with ProcessPoolExecutor(12, mp_context=mp.get_context('fork')) as ex:
    res = list(ex.map(run, designs, chunksize=8))
res.sort(key=lambda r: r[0])
for r in res[:20]:
    print(f'{r[0]:,.0f}', r[1], r[2])
json.dump([(r[0] if r[0] < math.inf else None, r[1], r[2]) for r in res[:300]], open('runs/search3.json', 'w'), indent=0, default=str)
