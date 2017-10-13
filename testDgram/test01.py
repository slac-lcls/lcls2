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
    classMembers=[]
    for item in inspect.getmembers(Dgram):
        classMembers.append(item[0])
    classMembers.sort()
    print("classMembers:", classMembers)

    d=Dgram(verbose, debug=debug)
    print("sys.getrefcount(d): %d\n" % sys.getrefcount(d))
    sys.stdout.flush()

    objAttributes=[]
    for item in inspect.getmembers(d):
        if item[0] not in classMembers:
            objAttributes.append(item[0])
    objAttributes.sort()

    for at in objAttributes:
        a1=getattr(d, at)
        sys.stdout.write("sys.getrefcount(%s) (refcount=%d):\n %s\n\n" %
                         (at, sys.getrefcount(a1), a1))
        a2=getattr(d, at)
        sys.stdout.write("sys.getrefcount(%s) (refcount=%d):\n %s\n\n" %
                         (at, sys.getrefcount(a2), a2))

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

