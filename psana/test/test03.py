#!/usr/bin/env python
#

import sys, os
import time
import getopt
import pprint

import inspect;
import gc

import pydgram

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
    print("gc.isenabled(): %s" % gc.isenabled())
    print("gc.get_debug(): %s" % gc.get_debug())
    print("gc.garbage:")
    print(gc.garbage)
    print("gc.get_count():")
    print(gc.get_count())
    print()

    print("get d1")
    fd = os.open(xtcdata_filename, os.O_RDONLY|os.O_LARGEFILE)
    d1=pydgram.PyDgram(fd, verbose=verbose, debug=debug)
    print(dir(d1))
    getAttr(d1, "d.array0_pgp")
    print("gc.garbage:")
    print(gc.garbage)
    print("gc.get_count():")
    print(gc.get_count())
    print()

    print("get d2")
    fd = os.open(xtcdata_filename, os.O_RDONLY|os.O_LARGEFILE)
    d2=pydgram.PyDgram(fd, verbose=verbose, debug=debug)
    getAttr(d2, "d.array0_pgp")
    print("gc.garbage:")
    print(gc.garbage)
    print("gc.get_count():")
    print(gc.get_count())
    print()

    print("del d1")
    del d1
    gc.collect()
    print("gc.garbage:")
    print(gc.garbage)
    print("gc.get_count():")
    print(gc.get_count())
    print()

    print("get d3")
    fd = os.open(xtcdata_filename, os.O_RDONLY|os.O_LARGEFILE)
    d3=pydgram.PyDgram(fd, verbose=verbose, debug=debug)
    getAttr(d3, "d.array0_pgp")
    print("gc.garbage:")
    print(gc.garbage)
    print("gc.get_count():")
    print(gc.get_count())
    print()


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
    gc.collect()
    time.sleep(5)
    print("gc.garbage:")
    print(gc.garbage)

    return

def usage_error():
    s="usage: python %s" %  os.path.basename(sys.argv[0])
    sys.stdout.write("%s [-h] [-v] [-d <DEBUG_NUMBER>]\n" % s)
    sys.exit(1)

if __name__=='__main__':
    main()

