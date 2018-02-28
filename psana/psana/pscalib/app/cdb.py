#------------------------------
"""
Created on 2018-02-23 by Mikhail Dubrovin
"""
#------------------------------

import sys
#from psana.pyalgos.generic.Utils import print_parser #print_kwargs
import psana.pscalib.calib.CalibConstants as cc
from psana.pscalib.calib.MDB_CLI import cdb

#------------------------------

def usage():
    return 'command examples\n'\
           '  cdb print -e cxi12345 -d camera-0-cxids1-0 -c pedestals -r 123 \n'\
           '  cdb save  -e cxi12345 -d camera-0-cxids1-0 -c pedestals -r 123 -f my_constants_save.txt\n'\
           '  cdb get   -e cxi12345 -d camera-0-cxids1-0 -c pedestals -r 123 -f my_constants_get.txt\n'\
           '  cdb -h'

#------------------------------

def cdb_cli() :
    """Calibration Data Base Command Line Interface
    """
    parser = input_option_parser()

    if len(sys.argv) == 1 : 
        print(80*'_')
        parser.print_help()
        print(80*'_')
        parser.print_usage()
        print(80*'_')
        msg = 'WARNING: COMMAND WITH ALL DEFAULT PARAMETERS IS USELESS...'
        #print(msg)
        sys.exit(msg)

    #print_parser(parser)
    cdb(parser)

    #(popts, pargs) = parser.parse_args()
    #opts = vars(popts)
    #kwargs = opts
    #print_kwargs(kwargs)
    #cdb(**kwargs)

#------------------------------

def input_option_parser() :

    from optparse import OptionParser

    d_host       = cc.HOST
    d_port       = cc.PORT
    d_experiment = 'cxi12345'
    d_detector   = 'camera-0-cxids1-0'
    d_ctype      = cc.list_calib_names[0]
    d_run        = 0
    d_time_stamp = '2008-01-01T00:00:00-0800'
    d_version    = 'v0'
    d_verbose    = True
    d_iofname    = './fname.txt'

    #d_evskip = 0
    #d_events = 1000
 
    h_host       = 'DB host, default = %s' % d_host
    h_port       = 'DB port, default = %s' % d_port
    h_experiment = 'experiment name, default = %s' % d_experiment 
    h_detector   = 'detector name, default = %s' % d_detector
    h_ctype      = 'calibration constant type, default = %s' % d_ctype 
    h_run        = 'run number, default = %d' % d_run 
    h_time_stamp = 'time stamp, default = %s' % d_time_stamp 
    h_version    = 'version of constants, default = %s' % d_version
    h_verbose    = 'verbosity, default = %s' % d_verbose
    h_iofname    = 'output file prefix, default = %s' % d_iofname

    #h_evskip     = 'number of events to skip before start processing, default = %s' % d_evskip
    #h_events     = 'number of events to process, default = %s' % d_events

    parser = OptionParser(description='Process hexanode xtc data and creates "small data" hdf5', usage=usage())

    parser.add_option('--host',             default=d_host,       action='store', type='string', help=h_host)
    parser.add_option('--port',             default=d_port,       action='store', type='string', help=h_port)
    parser.add_option('-d', '--detector',   default=d_detector,   action='store', type='string', help=h_detector)
    parser.add_option('-e', '--experiment', default=d_experiment, action='store', type='string', help=h_experiment)
    parser.add_option('-t', '--time_stamp', default=d_time_stamp, action='store', type='string', help=h_time_stamp)
    parser.add_option('-c', '--ctype',      default=d_ctype,      action='store', type='string', help=h_ctype)
    parser.add_option('-r', '--run',        default=d_run,        action='store', type='int',    help=h_run)
    parser.add_option('-v', '--version',    default=d_version,    action='store', type='string', help=h_version)
    parser.add_option('-p', '--verbose',    default=d_verbose,    action='store_false',          help=h_verbose)
    parser.add_option('-f', '--iofname',    default=d_iofname,    action='store', type='string', help=h_iofname)

    #parser.add_option('-m', '--evskip', default=d_evskip, action='store', type='int',    help=h_evskip)
    #parser.add_option('-n', '--events', default=d_events, action='store', type='int',    help=h_events)

    return parser
  
#------------------------------

if __name__ == "__main__" :
    cdb_cli()
    sys.exit(0)

#------------------------------
