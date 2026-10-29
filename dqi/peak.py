"""Find where the qubit peak happens: proxy the builder and record the call stack at the max."""
import sys, traceback; sys.path.insert(0, 'dqi')
import kickmix as km
mod = __import__(sys.argv[1])
real = km.CircuitBuilder


class Proxy:
    def __init__(self):
        self._b = real()
        self.peak = 0; self.where = None; self.hist = {}

    def __getattr__(self, name):
        attr = getattr(self._b, name)
        if not callable(attr):
            return attr

        def wrap(*a, **k):
            r = attr(*a, **k)
            n = self._b.num_allocated_qubits
            if n >= self.peak:
                st = [f'{f.name}:{f.lineno}' for f in traceback.extract_stack()[:-1] if 'dqi' in f.filename][-6:]
                if n > self.peak:
                    self.hist = {}
                self.peak = n
                key = ' > '.join(st)
                self.hist[key] = self.hist.get(key, 0) + 1
            return r
        return wrap


km.CircuitBuilder = Proxy
import gfq
p = {}
orig_build = mod.build
b_holder = {}
class Capture(Proxy):
    pass
km.CircuitBuilder = lambda: b_holder.setdefault('b', Proxy())
mod.km.CircuitBuilder = km.CircuitBuilder
c = mod.build()
b = b_holder['b']
print('peak', b.peak, 'num_qubits', c.num_qubits)
for k, v in sorted(b.hist.items(), key=lambda t: -t[1])[:6]:
    print(v, k)
