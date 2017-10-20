#!/usr/bin/env python
#

import sys, os
import time
import getopt
import pprint

import inspect;
import gc

sys.path.append('../build/xtcdata')
import dgram
#

class PyDgram:
    def __init__(self, xtcdata_filename, verbose=0, debug=0):
        self.verbose=verbose
        self.debug=debug

        f = os.open(xtcdata_filename, os.O_RDONLY|os.O_LARGEFILE)
        d=dgram.Dgram(f, verbose, debug)

        for key in sorted(d.__dict__.keys()):
             self.__dict__[key]=getattr(d, key)


def parse_command_line():
    opts, args_proper = getopt.getopt(sys.argv[1:], 'hvd:')
    verbose=0
    debug=0
    xtcdata_filename=None
    for option, parameter in opts:
        if option=='-h': usage_error()
        if option=='-v': verbose+=1
        if option=='-d': debug = int(parameter)
    if xtcdata_filename is None:
        xtcdata_filename="data.xtc"
    if verbose>0:
        sys.stdout.write("xtcdata filename: %s\n" % xtcdata_filename)
        sys.stdout.write("verbose: %d\n" % verbose)
        sys.stdout.write("debug: %d\n" % debug)
    elif debug>0:
        sys.stdout.write("debug: %d\n" % debug)
    return (args_proper, xtcdata_filename, verbose, debug)

def main():
    args_proper, xtcdata_filename, verbose, debug = parse_command_line()

    pd=PyDgram(xtcdata_filename, verbose, debug)

#    keys=sorted(d.__dict__.keys())
#    for key in keys:

    #print("pd.__dict__:")
    #pprint.pprint(pd.__dict__)
    #sys.stdout.flush()
    #print("pd.array0B:")
    #pprint.pprint(pd.array0B)
    #sys.stdout.flush()


#    print("pd.dgram_dict.keys(): %s" % pd.dgram_dict.keys())

#    for key in pd.dgram_dict.keys():
#        print("key: %s" % key)
#        pprint.pprint("     %s" % pd.dgram_dict[key])

    return

def usage_error():
    s="usage: python %s" %  os.path.basename(sys.argv[0])
    sys.stdout.write("%s [-h] [-v] [-d <DEBUG_NUMBER>]\n" % s)
    sys.exit(1)

if __name__=='__main__':
    main()

