#!/bin/env python

import argparse
from uuid import uuid4, uuid5, NAMESPACE_DNS

def generate(namespace, count):
    for x in range(count):
        timestamp = "{:08x}".format(x)
        msgid = uuid5(namespace, timestamp)
        print('%s' % msgid)
        
if __name__ == '__main__':

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--uuid5', action='store_true', help='use uuid5 for alloc namespace (default=uuid4)')
    parser.add_argument('-m', type=int, default=1, help='number of allocs (default=1)')
    parser.add_argument('-n', type=int, default=1, help='number of config cycles per alloc (default=1)')
    parser.add_argument('-q', action='store_true', help='quiet: do not print headers')
    args = parser.parse_args()
    quiet = args.q

    for alloc in range(args.m):
        if not quiet:
            print('# alloc %d of %d' % (alloc+1, args.m))

        if args.uuid5:
            ns_alloc = uuid5(NAMESPACE_DNS, 'slac.stanford.edu')
            if not quiet:
                print('# alloc namespace (uuid5): %s' % ns_alloc)
        else:
            ns_alloc = uuid4()
            if not quiet:
                print('# alloc namespace (uuid4): %s' % ns_alloc)

        ns_configure = uuid5(ns_alloc, 'configure')
        if not quiet:
            print('# configure namespace (uuid5): %s' % ns_configure)
            print('# %d configure IDs (uuid5):' % args.n)
        generate(ns_configure, args.n)

        ns_unconfigure = uuid5(ns_alloc, 'unconfigure')
        if not quiet:
            print('# unconfigure namespace (uuid5): %s' % ns_unconfigure)
            print('# %d unconfigure IDs (uuid5):' % args.n)
        generate(ns_unconfigure, args.n)
