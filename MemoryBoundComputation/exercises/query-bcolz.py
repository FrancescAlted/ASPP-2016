# Benchmark to compare the times for evaluating queries.
# Numexpr is needed in order to execute this.

import math
from time import time

import numpy as np
import numexpr as ne

import bcolz


N = int(1e7)  # the number of elements in x
clevel = 5  # the compression level
cname = "blosclz"  # the compressor name
#cname = "lz4"     # you may want to try this one too
sexpr = "(2*x*x + .3*y*y + z + 1) < 100"

print("Creating inputs...")

x = np.arange(N)
y = np.linspace(1, 10, N)
z = np.arange(N) * 10

# Build a ctable making use of above arrays as columns
cparams = bcolz.cparams(clevel=clevel, cname=cname)
t = bcolz.ctable((x, y, z, x * 2, y + .5, z / 10.),
                 names=['x', 'y', 'z', 'xp', 'yp', 'zp'],
                 cparams=cparams)
# The NumPy structured array version
nt = t[:]

print("Querying '%s' with 10^%d points" % (sexpr, int(math.log10(N))))

t0 = time()
out = [r for r in t[ne.evaluate(sexpr)]]
print("Time for structured array-->  *** %.3fs ***" % (time() - t0,))
print("out-->", len(out), out[:10])

# Uncomment the next for disabling threading
#ne.set_num_threads(1)
#bcolz.set_nthreads(1)

t0 = time()
#cout = t[t.eval(sexpr, cparams=cparams)]
cout = [r for r in t.where(sexpr)]
#cout = [r['x'] for r in t.where(sexpr)]
#cout = [r['y'] for r in t.where(sexpr, colnames=['x', 'y'])]
print("Time for ctable--> *** %.3fs ***" % (time() - t0,))
print("cout-->", len(cout), cout[:10])

#assert_array_equal(out, cout, "Arrays are not equal")

print("ctable:", repr(t))
