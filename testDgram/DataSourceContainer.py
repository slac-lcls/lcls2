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

class DataSource:
    def __init__(self, xtcdata_filename, verbose=0, debug=0):
        self.verbose=verbose
        self.debug=debug
        self.fd = os.open(xtcdata_filename,
                          os.O_RDONLY|os.O_LARGEFILE)
        self.config = dgram.Dgram(self.fd,
                                  verbose=verbose,
                                  debug=debug)

    def events(self):
        d=dgram.Dgram(self.fd, self.config,
                      verbose=self.verbose,
                      debug=self.debug)
        evt={}
        for key in sorted(d.__dict__.keys()):
            evt[key]=getattr(d, key)
        del(d)
        return evt

    def __iter__(self):
        return self

    def __next__(self):
        d=dgram.Dgram(self.fd, self.config,
                      verbose=self.verbose,
                      debug=self.debug)
        for key in sorted(d.__dict__.keys()):
            setattr(self, key, getattr(d, key))
        return self



def parse_command_line():
    opts, args_proper = getopt.getopt(sys.argv[1:], 'hvd:f:')
    verbose=0
    debug=0
    xtcdata_filename="data.xtc"
    for option, parameter in opts:
        if option=='-h': usage_error()
        if option=='-v': verbose+=1
        if option=='-d': debug = int(parameter)
        if option=='-f': xtcdata_filename = parameter
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

    ds=DataSource(xtcdata_filename, verbose=verbose, debug=debug)
    print("ds.__dict__:", ds.__dict__)

    count=0
    for evt in ds:
        print("\ncount: %d" % count)
        print("evt.__dict__:")
        pprint.pprint(evt.__dict__)
        count=+1
        sys.stdout.flush()

    return

def usage_error():
    s="usage: python %s" %  os.path.basename(sys.argv[0])
    sys.stdout.write("%s [-h] [-v] [-d <DEBUG_NUMBER>]\n" % s)
    sys.stdout.write("%s [-f xtcdata_filename]\n" % (" "*len(s)))
    sys.exit(1)

if __name__=='__main__':
    main()

