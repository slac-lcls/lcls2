import argparse
import sys
import itertools
import numpy as np
from collections import Counter

def sub_rates(factor):
    factors = [2,2,2,2,5,5,5,5,7,13]  # product is 910,000
    subfactors = []
    remfactors = []
    sf = factor
    for f in factors:
        d = sf/f
        if int(d)==d:
            sf = d
            remfactors.append(f)
        else:
            subfactors.append(f)

    rf = tuple(remfactors)

    iters = [itertools.combinations(subfactors,i+1) for i in range(len(subfactors))]
    f = set()
    d = {}
    ac = {}
    for i in iters:
        for c in i:
            q = np.prod(np.array(c))*factor
            f.add(q)
            d[q] = c+rf
            ac[q] = q if 13 not in d[q] else q/13

    f.add(factor)
    d[factor] = rf if len(rf) else (1,)

    base = 1300.e6/1400.
    result = []
    for q in sorted(f):
        result.append((int(base/float(q)),int(q/factor),q,d[q]))
    return result

def main():
    parser = argparse.ArgumentParser(description='simple validation printing')
    parser.add_argument("-f", "--factor", required=False , type=int, default=1, help="only show Nth sub-harmonics")
    parser.add_argument("-a", "--ac", action='store_true', help="format for AC coincidence tabl")
    parser.add_argument("-x", "--fixed", action='store_true', help="format for fixed rate coincidence tabl")
    args = parser.parse_args()

    result = sub_rates(args.factor)

    print('rate, Hz\tsubfactor\tfactor\tfactors')
    for q in sorted(result):
        print('{:8d}\t{:9d}\t{:6d}\t{}'.format(q[0],q[1],q[2],q[3]))

if __name__ == '__main__':
    main()
