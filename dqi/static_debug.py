import sys; sys.path.insert(0, 'dqi')
import numpy as np
import gfalg as G
from divstep_model import mul, inv, poly_eval, n
import static_model as SM

rng = np.random.default_rng(5)
w = 33
pos = rng.choice(4095, w, replace=False)
err = {int(p): int(rng.integers(1, 4096)) for p in pos}
S = [0] * n
for j, v in err.items():
    for i in range(n):
        S[i] ^= mul(v, G.gpow(2, ((i + 1) * j) % 4095))
# list model
f = [1]; g = S[::-1]; bf = [0]; bg = [1]; pat = ''
for step in range(70):
    sw = step % 2 == 0 and g[0] != 0
    if sw: f, g, bf, bg = g, f, bg, bf
    c = mul(g[0], inv(f[0])); pat += 'S' if sw else ('n' if c else '0')
    L = max(len(f), len(g)); f += [0] * (L - len(f)); g += [0] * (L - len(g))
    g = [g[i] ^ mul(c, f[i]) for i in range(L)][1:]
    Lb = max(len(bf), len(bg)); bf += [0] * (Lb - len(bf)); bg += [0] * (Lb - len(bg))
    bg = [bg[i] ^ mul(c, bf[i]) for i in range(Lb)]
    bf = [0] + bf
print(pat)
print('list model: bg roots ok', all(poly_eval(bg, G.gpow(2, j)) == 0 for j in err))
P = [70, 70, 70, 69] + [max(37, 70 - (s - 1) // 2) for s in range(4, 70)]
F, Gr, ok = SM.forward(S, P)
print('static ok', ok)
bgs = [F[SM.K - j] for j in range(len(bg))]
print('bg equal', bgs[:40] == (bg + [0] * 40)[:40])
gs = Gr[:len(g)]
print('g equal', gs == g, len(g))
print('decode ok', SM.decode(F, Gr) == err)

# lockstep
N, K = SM.N, SM.K
Fa = [0] * N; Ga = [0] * N; Fa[0] = 1; Fa[K] = 1
for i in range(70): Ga[i] = S[69 - i]
f = [1]; g = S[::-1]; bf = [0]; bg = [1]
for step in range(12):
    if step % 2 == 0 and g[0] != 0:
        f, g, bf, bg = g, f, bg, bf
    if step % 2 == 0 and Ga[0] != 0:
        Fa, Ga = Ga, Fa
    c = mul(g[0], inv(f[0])); ca = mul(Ga[0], inv(Fa[0]))
    L = max(len(f), len(g)); f += [0] * (L - len(f)); g += [0] * (L - len(g))
    g = [g[i] ^ mul(c, f[i]) for i in range(L)][1:]
    Lb = max(len(bf), len(bg)); bf += [0] * (Lb - len(bf)); bg += [0] * (Lb - len(bg))
    bg = [bg[i] ^ mul(c, bf[i]) for i in range(Lb)]
    bf = [0] + bf
    p = P[step]
    for i in range(N):
        if i < p: Ga[i] ^= mul(ca, Fa[i])
        else: Fa[i] ^= mul(ca, Ga[i])
    Ga = Ga[1:] + [0]
    fa = Fa[:len(f)]; ga = Ga[:len(g)]
    bga = [Fa[K - j] for j in range(len(bg))]; bfa = [Ga[K - j] for j in range(len(bf))]
    print(step, c == ca, fa == f, ga == g, bga == bg, bfa == bf, len(f), len(g), len(bf), len(bg))
