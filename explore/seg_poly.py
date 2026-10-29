"""Ước lượng: xấp xỉ 1/sqrt(M), M in [1,4), bằng đa thức bậc d trên 2^k đoạn đều (minimax qua Chebyshev).
In ra sai số tương đối lớn nhất (chưa tính làm tròn)."""
import numpy as np
from numpy.polynomial import chebyshev as C
for lo, hi in ((1.0, 4.0), (1.0, 2.0)):
    for d in (1, 2, 3, 4):
        for k in range(2, 12):
            nseg = 2**k
            edges = np.linspace(lo, hi, nseg + 1)
            worst = 0
            for a, b in zip(edges[:-1], edges[1:]):
                t = np.linspace(a, b, 400)
                f = 1/np.sqrt(t)
                cf = C.chebfit((2*t-a-b)/(b-a), f, d)
                err = np.abs(C.chebval((2*t-a-b)/(b-a), cf) - f) * np.sqrt(t)
                worst = max(worst, err.max())
            if worst < 2**-26:
                print(f'range[{lo},{hi}) deg {d}: {nseg:5d} segments -> rel err 2^{np.log2(worst):.1f}')
                break
