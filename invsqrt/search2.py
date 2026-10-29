import sys, itertools, json, math
sys.path.insert(0, 'invsqrt')
from concurrent.futures import ProcessPoolExecutor
import gen_poly2 as G

def run(d):
    try:
        s, info = G.estimate(d)
    except Exception as ex:
        return (math.inf, str(d), repr(ex)[:80])
    return (s, str(d), info)

designs = []
for k, gb, use_and in itertools.product((6, 7, 8), (26, 27, 28), (True, False)):
    for t1 in range(gb - 9, gb - 4):
        for t2 in range(t1 - 11, t1 - 6):
            for g0, g1 in ((2, 1), (3, 2), (4, 2)):
                for d0 in range(21 - k - 3, 22 - k):
                    for d1 in range(max(3, t2 - k - 2), min(22 - k, t2 - k + 4)):
                        designs.append(G.Design(k=k, degree=2, g_bits=gb, t_bits=(t1, t2), guard=(g0, g1),
                                                dbits=(d0, d1), use_and=use_and))
print(len(designs), flush=True)
with ProcessPoolExecutor(16) as ex:
    res = list(ex.map(run, designs, chunksize=4))
res.sort(key=lambda r: r[0])
for r in res[:15]:
    print(f'{r[0]:,.0f}', r[1], r[2])
json.dump([(r[0] if r[0] < math.inf else None, r[1], r[2]) for r in res[:200]], open('runs/search2.json', 'w'), indent=0, default=str)
