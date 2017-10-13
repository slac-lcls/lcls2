#
#
#
 
import sys, os
import getopt

sys.path.append('../build/xtcdata')
from dgram import Dgram
#

def do_it(args_proper, verbose, debug):
    d=Dgram(verbose, debug=debug)
    dir(d)
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
    if debug>0:
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

