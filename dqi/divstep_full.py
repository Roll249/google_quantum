"""Untruncated divstep model with the packed static layout F=(f,bg), G=(g,bf); find Omega for Forney."""
import sys; sys.path.insert(0, 'dqi')
import numpy as np
import gfalg as G
from divstep_model import mul, inv, syndromes, poly_eval, n, m, ell

N = int(sys.argv[2]) if len(sys.argv) > 2 else 72
K = N - 1


def run(S, check=None):
    F = [0] * N; Gr = [0] * N
    F[0] = 1          # f = 1 (reversed x^70)
    F[K] = 1          # bg_0 = 1
    for i in range(70):
        Gr[i] = S[69 - i]   # g = rev(S)
    info = []
    for pair in range(35):
        for half in range(2):
            stepn = 2 * pair + half
            if half == 0 and Gr[0] != 0:
                F, Gr = Gr, F
            c = mul(Gr[0], inv(F[0]))
            # lengths for the split
            lf = max([i + 1 for i in range(N) if F[i]] and [0])
            info.append(c)
            P = check(stepn) if check else None
            yield ('pre', stepn, F[:], Gr[:], c)
            newF, newG = F[:], Gr[:]
            for i in range(N):
                if i < P:
                    newG[i] ^= mul(c, F[i])
                else:
                    newF[i] ^= mul(c, Gr[i])
            assert newG[0] == 0
            F, Gr = newF, newG[1:] + [0]
    yield ('end', 70, F, Gr, None)
