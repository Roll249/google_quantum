import sys, json, math, multiprocessing as mp
sys.path.insert(0, 'invsqrt')
from concurrent.futures import ProcessPoolExecutor
import gen_poly3 as G
from gen_poly3 import Design
def run(s):
    d = eval(s)
    try:
        sc, info = G.estimate(d)
    except Exception as ex:
        return (math.inf, s, repr(ex)[:80])
    return (sc, s, info)
rows = json.load(open(sys.argv[1]))
cands = [r[1] for r in rows if r[0] is not None]
with ProcessPoolExecutor(14, mp_context=mp.get_context('fork')) as ex:
    res = sorted(ex.map(run, cands, chunksize=4), key=lambda r: r[0])
for r in res[:10]:
    print(f'{r[0]:,.0f}', r[1], r[2])
json.dump(res[:50], open('runs/rerank3.json', 'w'), default=str, indent=0)
