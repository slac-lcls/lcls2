#!/usr/bin/env python3

"""Simple shared-memory client: read the events of a psana shmem DataSource and count them."""
import argparse
from psana import DataSource

def shmemClientSimple(args):
    """Read all events of the first run of `DataSource(shmem=args.partitionTag)` and return their number.

    With `args.verbose` each event's service code, timestamp (hex and seconds.nanoseconds) and
    `evt._size` are printed.
    """
    dg_count = 0
    ds = DataSource(shmem=args.partitionTag)
    run = next(ds.runs())
    for evt in run.events():
        if args.verbose:
            print('%-15d transition: time 0x%016x = %u.%09u, payloadSize 0x%x' % (evt.service(), evt.timestamp, evt._seconds, evt._nanoseconds, evt._size))
        dg_count += 1
    return dg_count

#------------------------------

def main() :
    """Parse -p/--partitionTag (default 'tst') and -v/--verbose and run `shmemClientSimple` with them."""
    hutch = 'tst'

    parser = argparse.ArgumentParser(description='Python shmemClient')
    parser.add_argument('-p', '--partitionTag', help='partitionTag ['+hutch+']', type=str, default=hutch)
    parser.add_argument('-v', '--verbose',      help='verbose flag', action='store_true')

    shmemClientSimple(parser.parse_args())

#------------------------------

if __name__ == '__main__':
    main()

#------------------------------
