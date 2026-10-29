import sys; sys.path.insert(0, 'dqi'); sys.path.insert(0, 'data/verifiers/dqi')
import numpy as np, multiprocessing as mp
import gfalg as G
from divstep_model import mul, inv
import verify_submission_dqi as V

def work(seed, n=5000):
    rng = np.random.default_rng(seed)
    wp = V.optimal_dqi_weights(4095, 35, 12, 2016) ** 2
    exp, log = G.tables()
    kupd = np.full(70, -1); kdis = np.full(70, -1)
    hist = []          # per shot: per-round (deg sigma_new, deg tau~) when D != 0
    stack2 = 0
    rows = []
    for t in range(n):
        w = int(rng.choice(36, p=wp)); pos = rng.choice(4095, w, replace=False); vals = rng.integers(1, 4096, w)
        S = [0] * 70
        for j, v in zip(pos, vals):
            for i in range(70):
                S[i] ^= exp[(log[int(v)] + (i + 1) * int(j)) % 4095]
        sig = [1] + [0] * 35; tau = [1] + [0] * 35; d = 0; st = 0
        row = np.full((70, 2), -1)
        for r in range(1, 71):
            tau = [tau[-1]] + tau[:-1]
            dsig = max([i for i, v in enumerate(sig) if v])
            D = 0
            for i in range(0, min(r - 1, 35) + 1):
                D ^= mul(sig[i], S[r - 1 - i])
            rep = D != 0 and d >= 0; ret = D != 0 and d < 0
            if D:
                p = mul(D, sig[0]); m = inv(p) if rep else p
                if rep: sig, tau = tau, sig
                sig = [sig[i] ^ mul(m, tau[i]) for i in range(36)]
                row[r - 1] = (max(max([i for i, v in enumerate(sig) if v]), max([i for i, v in enumerate(tau) if v])), dsig)
            st += ret
            d = -d if rep else d + 1
        rows.append(row)
        stack2 += st >= 2
    return np.array(rows), stack2

if __name__ == '__main__':
    with mp.get_context('fork').Pool(12) as pool:
        res = pool.map(work, range(100, 112))
    R = np.concatenate([r[0] for r in res]); s2 = sum(r[1] for r in res)
    np.save('runs/bm_rows.npy', R)
    print('shots', len(R), 'stack>=2', s2)
    for q in (100, 99.99, 99.9):
        print(q, 'upd', np.percentile(R[:, :, 0], q, axis=0).astype(int).tolist())
    print('dis max', R[:, :, 1].max(axis=0).tolist())
