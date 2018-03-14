#------------------------------
"""
Created on 2018-02-23 by Mikhail Dubrovin
"""
#------------------------------

import sys
#from psana.pyalgos.generic.Utils import print_parser # print_kwargs
import psana.pscalib.calib.CalibConstants as cc
from psana.pscalib.calib.MDB_CLI import cdb

#------------------------------

def usage():
    return 'command examples\n'\
           '  cdb print\n'\
           '  cdb print --host psanaphi105 --port 27017\n'\
           '  cdb print -e cxi12345\n'\
           '  cdb print -d camera-0-cxids1-0\n'\
           '  cdb print -e cxi12345 -d camera-0-cxids1-0 -c pedestals -r 123 \n'\
           '  cdb add   -e cxi12345 -d camera-0-cxids1-0 -c pedestals -r 123 -f my_constants_save.txt\n'\
           '  cdb get   -e cxi12345 -d camera-0-cxids1-0 -c pedestals -r 123 -f my_constants_get.txt\n'\
           '  cdb convert -e cxix25615 -p\n'\
           '  cdb delete  -e cxix25615\n'\
           '  cdb delete --dbname cxix25615\n'\
           '  cdb export --dbname cxix25615\n'\
           '  cdb import --dbname cxix25615 --iofname cdb-...arc\n'\
           '  cdb get -e cxif5315 -d cxids2-0-cspad-0 -c geometry -r 1 -f geo.txt -p\n'\
           '  cdb get -e cxix25115 -d cxidg3-0-opal1000-0 -c pedestals -r 6 -f nda.npy -p\n'\
           '  cdb -h\n'\
           '  cdb\n'\
           '  \nTEST:\n'\
           '  cdb print -e exp12345 -d camera-1 -c pedestals -r 123 -p\n'\
           '  cdb add   -e exp12345 -d camera-1 -c pedestals -r 123 -f 6-end.data -p\n'\
           '  cdb get   -e exp12345 -d camera-1 -c pedestals -r 123 -f nda.npy -p\n'\
           '  etc.'\

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
    d_dbname     = None
    d_experiment = None
    d_detector   = None
    d_ctype      = cc.list_calib_names[0]
    d_run        = '0'
    d_run_end    = 'end'
    d_time_stamp = None # '2001-09-08T18:46:40-0700'
    d_time_sec   = None # '1000000000'
    d_version    = 'v0'
    d_verbose    = False
    d_iofname    = None # './fname.txt'
    d_comment    = 'No comment'

    h_host       = 'DB host, default = %s' % d_host
    h_port       = 'DB port, default = %s' % d_port
    h_dbname     = 'database name for direct command like "delete", default = %s' % d_dbname
    h_experiment = 'experiment name, default = %s' % d_experiment 
    h_detector   = 'detector name, default = %s' % d_detector
    h_ctype      = 'calibration constant type, default = %s' % d_ctype 
    h_run        = 'run number (begin), default = %s' % d_run 
    h_run_end    = 'run number (end), default = %s' % d_run_end
    h_time_stamp = 'time stamp, default = %s' % d_time_stamp 
    h_time_sec   = 'time (sec), default = %s' % d_time_sec
    h_version    = 'version of constants, default = %s' % d_version
    h_verbose    = 'verbosity, default = %s' % d_verbose
    h_iofname    = 'output file prefix, default = %s' % d_iofname
    h_comment    = 'comment to the document, default = %s' % d_comment

    parser = OptionParser(description='Command line interface to LCLS2 calibration data base', usage=usage())

    parser.add_option('--host',             default=d_host,       action='store', type='string', help=h_host)
    parser.add_option('--port',             default=d_port,       action='store', type='string', help=h_port)
    parser.add_option('--dbname',           default=d_dbname,     action='store', type='string', help=h_dbname)
    parser.add_option('-d', '--detector',   default=d_detector,   action='store', type='string', help=h_detector)
    parser.add_option('-e', '--experiment', default=d_experiment, action='store', type='string', help=h_experiment)
    parser.add_option('-t', '--time_stamp', default=d_time_stamp, action='store', type='string', help=h_time_stamp)
    parser.add_option('-s', '--time_sec',   default=d_time_sec,   action='store', type='string', help=h_time_sec)
    parser.add_option('-c', '--ctype',      default=d_ctype,      action='store', type='string', help=h_ctype)
    parser.add_option('-r', '--run',        default=d_run,        action='store', type='string', help=h_run)
    parser.add_option('-u', '--run_end',    default=d_run_end,    action='store', type='string', help=h_run_end)
    parser.add_option('-v', '--version',    default=d_version,    action='store', type='string', help=h_version)
    parser.add_option('-p', '--verbose',    default=d_verbose,    action='store_true',           help=h_verbose)
    parser.add_option('-f', '--iofname',    default=d_iofname,    action='store', type='string', help=h_iofname)
    parser.add_option('-m', '--comment',    default=d_comment,    action='store', type='string', help=h_comment)

    return parser
  
#------------------------------

if __name__ == "__main__" :
    cdb_cli()
    sys.exit(0)

#------------------------------
