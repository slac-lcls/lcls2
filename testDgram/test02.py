#!/usr/bin/env python
#

import sys, os
import time
import getopt

import inspect;
#from gc import get_referrers

sys.path.append('../build/xtcdata')
from dgram import Dgram
#

def setAttr(d, attr, value, verbose=None):
    if verbose is not None:
        verbose0=d.verbose
        d.verbose=verbose
    print("Set value of %s:" % (attr))
    setattr(d, attr, value)
    print(value)
    if verbose is not None:
        d.verbose=verbose0
    return True

def getAttr(d, attr, verbose=None):
    if verbose is not None:
        verbose0=d.verbose
        d.verbose=verbose
    print("Get value of %s (id=%s):" % (attr, id(attr)))
    value=getattr(d, attr)
    print(value)
    if verbose is not None:
        d.verbose=verbose0
    return value

def do_it(args_proper, xtcdata_filename, verbose, debug):
    fd = os.open(xtcdata_filename, os.O_RDONLY|os.O_LARGEFILE)
    d=Dgram(fd, verbose, debug)
    print("d:", d)
    print("id(d):", id(d))
 
    a1=getAttr(d, 'array0')
    print("a1.base:", a1.base)
    print("id(a1):", id(a1))
    print("dir(a1):")
    dir(a1)
#
#    d.verbose=1
#    a2=getAttr(d, 'array0')
#    print("id(a2):", id(a2))
#    print("del a2")
#    del a2
#
#    a2=getAttr(d, 'array0')
#    print("id(a2):", id(a2))
#    print("del a2")
#    del a2
#
#    print("del d")
#    del d
#
#    print(a1)
#    del a1

def parse_command_line():
    opts, args_proper = getopt.getopt(sys.argv[1:], 'hvd:')
    verbose=0
    debug=0
    xtcdata_filename=None
    for option, parameter in opts:
        if option=='-h': usage_error()
        if option=='-v': verbose+=1
        if option=='-d': debug = int(parameter)
    if xtcdata_filename is None: xtcdata_filename="data.xtc"
    if verbose>0:
        sys.stdout.write("xtcdata filename: %s\n" % xtcdata_filename)
        sys.stdout.write("verbose: %d\n" % verbose)
        sys.stdout.write("debug: %d\n" % debug)
    elif debug>0:
        sys.stdout.write("debug: %d\n" % debug)
    return (args_proper, xtcdata_filename, verbose, debug)

def main():
    args_proper, xtcdata_filename, verbose, debug = parse_command_line()

    do_it(args_proper, xtcdata_filename, verbose, debug)

    return

def usage_error():
    s="usage: python %s" %  os.path.basename(sys.argv[0])
    sys.stdout.write("%s [-h] [-v] [-d <DEBUG_NUMBER>]\n" % s)
    sys.exit(1)

if __name__=='__main__':
    main()

