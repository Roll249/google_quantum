import sys, pickle; sys.path.insert(0, 'dqi')
sys.argv = [sys.argv[0], '0', '0', '74']
import general_model as GM, static_model as SM
import decoder3 as D3
from divstep_model import mul, inv
fails = pickle.load(open('runs/v3_fails.pkl', 'rb'))
K = 73
for err, S in fails:
    F, Gr, delta, ok, kinds = GM.forward(S, D3.P)
    wv = (71 - delta) // 2
    F2, G2 = SM.cleanup(F, Gr, wv)
    good = SM.chien(F2, G2) == err
    # split constraint check for this shot
    f = [1]; g = S[::-1]; bf = [0]; bg = [1]; d = 1
    dg = lambda p: max([i for i, v in enumerate(p) if v] or [-1])
    viol = []
    for step in range(70):
        if d > 0 and g[0] != 0:
            f, g, bf, bg, d = g, f, bg, bf, -d
        c = mul(g[0], inv(f[0]))
        if c:
            lo = max(dg(f), dg(g)) + 1; hi = K - max(dg(bf), dg(bg))
            if not (lo <= D3.P[step] <= hi): viol.append((step, lo, hi, D3.P[step]))
        L = max(len(f), len(g)); f += [0] * (L - len(f)); g += [0] * (L - len(g))
        g = [g[i] ^ mul(c, f[i]) for i in range(L)][1:]
        Lb = max(len(bf), len(bg)); bf += [0] * (Lb - len(bf)); bg += [0] * (Lb - len(bg))
        bg = [bg[i] ^ mul(c, bf[i]) for i in range(Lb)]
        bf = [0] + bf
        d += 1
    print(len(err), 'model ok', ok, 'decode', good, 'w', wv, 'viol', viol[:3])
