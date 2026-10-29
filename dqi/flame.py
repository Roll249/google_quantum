"""Hierarchical Toffoli breakdown from kickmix flame charts: path -> (count, toffoli)."""
import re, sys
import kickmix as km


def tree(c, max_depth=6):
    s = c.flame_chart_svg()
    tot = c.operation_statistics().num_toffoli_like
    items = re.findall(r'<rect x="([\d.]+)" y="(\d+)" width="([\d.]+)" height="10"[^>]*/>\s*<text[^>]*>([^<]+)</text>', s)
    rows = sorted((int(y) // 10, float(x), float(w), lab) for x, y, w, lab in items)
    out = []
    for d, x, w, lab in rows:
        parent = None
        for pd, px, pw, plab, path in reversed(out):
            if pd == d - 1 and px <= x + 1e-9 and x + w <= px + pw + 1e-6:
                parent = path
                break
        path = (parent + ' / ' if parent else '') + lab
        out.append((d, x, w, lab, path))
    agg = {}
    for d, x, w, lab, path in out:
        if d <= max_depth:
            agg[path] = agg.get(path, 0) + w * tot / 1000
    return tot, agg


if __name__ == '__main__':
    c = km.Circuit.from_file(sys.argv[1])
    tot, agg = tree(c)
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.003
    for k, v in agg.items():
        if v > thr * tot:
            print(f'{v:12.0f}  {k}')
    print('total', tot, 'qubits', c.num_qubits)
