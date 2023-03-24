import sys
import itertools
import numpy as np

if __name__ == '__main__':
    factors = [2,2,2,2,5,5,5,5,7,13]  # product is 910,000
    iters = [itertools.combinations(factors,i+1) for i in range(len(factors))]
    f = set()
    d = {}
    for i in iters:
        for c in i:
            q = np.prod(np.array(c))
            f.add(q)
            d[q] = c

    f.add(1)
    d[1] = 1

    base = 1300.e6/1400.
    print(' rate, Hz  | factor | factors')
    for q in sorted(f):
        print(' {:6d}     {:6d}   {}'.format(int(base/float(q)),q,d[q]))
        
