#!/usr/bin/env python
""" test argparse
"""

import sys
print('e.g.: [python] %s amox23616 104 --nevts 300 -c 123' % sys.argv[0])

import argparse

def test_argparse() :

    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str, help='psana experiment string (e.g. amox23616)')
    parser.add_argument('run', type=int, help='run number')
    parser.add_argument('--nevts', nargs='?', const=400, type=int, default=400)
    parser.add_argument('-c', type=float, help='coordinate', default=1.4, dest="cord")
    parser.add_argument('-a', type=float, help='area', default=2.2)
    #parser.add_argument('--range', nargs=2, const=None, type=tuple, default=None)

    args = parser.parse_args()
    print('parser.parse_args()', args)

    print('args.exp',   args.exp)
    print('args.run',   args.run)
    print('args.nevts', args.nevts)
    print('args.cord',  args.cord)

    d_args = vars(args)
    print('vars(args)', d_args)

    for k in d_args :
        print('defauld for key %8s :' % k, parser.get_default(k))

    #d_def = vars(parser.get_default())
    #print('vars(def)', d_def)


    for k in d_args :
        print('value for key %8s :' % k, d_args[k])

#----------

if __name__ == "__main__":
    test_argparse()

#----------
