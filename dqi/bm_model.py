"""Classical model of the rescaled reversible Berlekamp-Massey round (Gu et al. 2026, SI I.1)."""
import sys; sys.path.insert(0, 'dqi'); sys.path.insert(0, 'data/verifiers/dqi')
import numpy as np
import gfalg as G
from divstep_model import mul, inv, poly_eval

L = 36


def bm(S, track=False):
    """S[i] = syndrome S_{i+1}. Returns sigma (36 coeffs), stats."""
    sig = [1] + [0] * (L - 1)
    tau = [1] + [0] * (L - 1)
    d = 0
    stack = []
    degs = []
    for r in range(1, 71):
        tau = [tau[-1]] + tau[:-1]                # z * tau (cyclic relabel)
        D = 0
        for i in range(0, min(r - 1, L - 1) + 1):
            D ^= mul(sig[i], S[r - 1 - i])
        rep = D != 0 and d >= 0
        ret = D != 0 and d < 0
        p = mul(D, sig[0])
        m = inv(p) if rep else p
        if rep:
            sig, tau = tau, sig
        sig = [sig[i] ^ mul(m, tau[i]) for i in range(L)]
        if ret:
            stack.append((r, D))
        d = -d if rep else d + 1
        # reversibility facts: replace <=> tau_0 != 0 and d <= 0
        if rep:
            assert mul(sig[0], D) == 1 and tau[0] != 0 and d <= 0
        elif tau[0] != 0:
            assert d >= 1, (r, d)
        dg = lambda q: max([i for i, v in enumerate(q) if v] or [-1])
        degs.append((dg(sig), dg(tau) if D else -1))
    return sig, stack, degs


if __name__ == '__main__':
    import verify_submission_dqi as V
    rng = np.random.default_rng(int(sys.argv[1]))
    wp = V.optimal_dqi_weights(4095, 35, 12, 2016) ** 2
    exp, log = G.tables()
    Ns = int(sys.argv[2])
    maxdeg = np.zeros((70, 2), int); stackmax = []; bad = 0
    degsamples = []
    for t in range(Ns):
        w = int(rng.choice(36, p=wp)); pos = rng.choice(4095, w, replace=False); vals = rng.integers(1, 4096, w)
        S = [0] * 70
        for j, v in zip(pos, vals):
            for i in range(70):
                S[i] ^= exp[(log[int(v)] + (i + 1) * int(j)) % 4095]
        sig, stack, degs = bm(S)
        ok = all(poly_eval(sig, G.ginv(G.gpow(2, int(j)))) == 0 for j in pos) and max(i for i, v in enumerate(sig) if v) == w
        bad += not ok
        stackmax.append(len(stack))
        degsamples.append(degs)
    D = np.array(degsamples)  # Ns x 70 x 2
    print('decode failures', bad, 'stack sizes', np.bincount(stackmax))
    for q in (1.0,):
        mx = D.max(axis=0)
        print('max deg sigma per round', mx[:, 0].tolist())
        print('max deg tau per round  ', mx[:, 1].tolist())
    np.save('runs/bm_degs.npy', D)
