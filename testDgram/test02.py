#!/usr/bin/env python
#

import sys
import time
import getopt

import inspect;
#from gc import get_referrers

sys.path.append('../build/xtcdata')
from dgram import Dgram
#

def do_it(args_proper, verbose, debug):
    d=Dgram(verbose, debug=debug)

    a1=d.fexfloat1
    a2=d.fexfloat1
    a3=d.fexfloat1
    a4=d.fexfloat1

    print("del a1, a2, a3, a4")
    del a1, a2, a3, a4

    a1=d.fexfloat1
    a2=d.fexfloat1
    a3=d.fexfloat1
    a4=d.fexfloat1

    return True

def parse_command_line():
    opts, args_proper = getopt.getopt(sys.argv[1:], 'hvd:')
    verbose=0
    debug=0
    for option, parameter in opts:
        if option=='-h': usage_error()
        if option=='-v': verbose+=1
        if option=='-d': debug = int(parameter)
    if verbose>0:
        sys.stdout.write("verbose: %d\n" % verbose)
        sys.stdout.write("debug: %d\n" % debug)
    elif debug>0:
        sys.stdout.write("debug: %d\n" % debug)
    return (args_proper, verbose, debug)

def main():
    args_proper, verbose, debug = parse_command_line()

    do_it(args_proper, verbose, debug)

    return

def usage_error():
    s="usage: python %s" %  os.path.basename(sys.argv[0])
    sys.stdout.write("%s [-h] [-v] [-d <DEBUG_NUMBER>]\n" % s)
    sys.exit(1)

if __name__=='__main__':
    main()

