import sys
import argparse
import itertools
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='sequence pva programming')
    parser.add_argument("--ued", action='store_true', help="verbose output")
    args = parser.parse_args()

    if args.ued:
        factors = [2,2,2,2,2,5,5,5,5,5,5]  # product is 500,000
        base = 500.e3
    else:
#        factors = [2,2,2,2,5,5,5,5,7,13]  # product is 910,000 (13/14 MHz)
#        base = 1300.e6/(7*200)
#  Future considerations
        factors = [2,2,2,2,2,2,5,5,5,5,5,7]  # product is 1,400,000 (10/7 MHz)
        base = 1300.e6/(7*130)
#        factors = [2,2,2,2,2,5,5,5,5,5,7]  # product is 700,000 (5/7 MHz)
#        base = 1300.e6/(7*260)
#        factors = [2,2,2,2,2,2,2,5,5,5,5,7]  # product is 560,000 (4/7 MHz)
#        base = 1300.e6/(7*325)

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
