import sys
import argparse
import itertools
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Find all RF solutions')
    args = parser.parse_args()

    factors = [2,2,5,5,13] # product is 1300
    base = 1300.e6/7.

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

    print(' rate, Hz  | factor | factors')
    for q in sorted(f):
        print(' {:6d}     {:6d}   {}'.format(int(base/float(q)),q,d[q]))
        

if __name__ == '__main__':
    main()
