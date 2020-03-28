#------------------------------
"""
Created on 2018-02-23 by Mikhail Dubrovin
"""
#------------------------------

import sys

#from psana.pyalgos.generic.Utils import print_parser # print_kwargs
import psana.pscalib.calib.CalibConstants as cc
from psana.pscalib.calib.MDB_CLI import cdb, logging, MODES

LEVEL_NAMES = ', '.join(list(logging._levelToName.values()))

#------------------------------

def usage():
    return '\nCommand: cdb <mode> [options]'\
           '\n              modes: %s\n'%(', '.join(MODES))\
         + '\nExamples:\n'\
           '  cdb\n'\
           '  cdb -h\n'\
           '  cdb print\n'\
           '  cdb print -e exp12345\n'\
           '  cdb print -d detector_1234\n'\
           '  cdb print --dbname cdb_exp12345\n'\
           '  cdb print --dbname cdb_detector_1234\n'\
           '  cdb convert -e cxif5315 -u dubrovin -p <password>\n'\
           '  cdb convert -e amox23616 -u dubrovin -p <password>\n'\
           '  cdb get -e exp12345 -d detector_1234 -c testdict -r 23 -f mydict\n'\
           '  cdb get -e cxic0415 -d cspad_0001 -c pedestals -s 1520977960 -f mypeds\n'\
           '  cdb get -e cxic0415 -d cspad_0001 -c geometry -r 100 -f mygeo\n'\
           '  cdb get -d cspad_0001 -c pedestals -r 100 -f mypeds\n'\
           '  cdb add -e exp12345 -d detector_1234 -c pedestals -r 123 -f mypeds.data -u dubrovin\n'\
           '  cdb add -e new55555 -d detnew_5555   -c pedestals -r 123 -f mypeds.data -u dubrovin\n'\
           '  cdb add -e amox27716 -d tmo_quadanode -c calibcfg -r 100 -f configuration_quad.txt -i txt -u dubrovin\n'\
           '  cdb add -e amox23616 -d xtcav -c pedestals -r 104 -f xtcav_peds.data -i xtcav -u dubrovin\n'\
           '  cdb deldoc -e exp12345 -d detector_1234 -c pedestals -r 123 -v 05 -u <username> -p <password> -C\n'\
           '  cdb deldoc -e cxix25615 -d cspad_0001 -c pedestals -r 125 -u <username> -p <password> -C\n'\
           '  cdb deldoc -e cxix25615 -d cspad_0001 -c pedestals -s 1520977960 -u <username> -p <password> -C\n'\
           '  cdb delcol -e cxix25615 -d cspad_0001 -u <username> -p <password> -C\n'\
           '  cdb delcol -d cspad_0001 -u <username> -p <password> -C\n'\
           '  cdb deldb -e amox23616 -u <username> -p <password> -C\n'\
           '  cdb deldb -d opal1000_0059 -u <username> -p <password> -C\n'\
           '  cdb deldb --dbname cdb_amox23616 -u dubrovin -p <password> -C\n'\
           '  cdb delall\n'\
           '  cdb export --dbname cdb_exp12345\n'\
           '  cdb import --dbname cdb_exp12345 --iofname cdb-...arc\n'\
           '  cdb print --host=psanagpu115 --port=27017 --stout=1000'

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
        sys.exit('COMMAND WITHOUT PARAMETERS')

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
    d_user       = cc.USERNAME
    d_upwd       = ''
    d_ctout      = 5000
    d_stout      = 30000
    d_dbname     = None
    d_experiment = None
    d_detector   = None
    d_ctype      = None # cc.list_calib_names[0], 'pedestals'
    d_dtype      = None
    d_run        = None
    d_run_end    = None
    d_time_stamp = None # '2001-09-08T18:46:40-0700'
    d_time_sec   = None
    d_version    = None
    d_confirm    = False
    d_iofname    = None # './fname.txt'
    d_comment    = 'No comment'
    d_loglevel   = 'INFO'

    h_host       = 'DB host, default = %s' % d_host
    h_port       = 'DB port, default = %s' % d_port
    h_user       = 'username to access DB, default = %s' % d_user
    h_upwd       = 'password, default = %s' % d_upwd
    h_ctout      = 'connect timeout connectTimeoutMS, default = %d' % d_ctout
    h_stout      = 'socket timeout serverSelectionTimeoutMS, default = %d' % d_stout
    h_dbname     = 'database name, works for mode "print" or "delete", default = %s' % d_dbname
    h_experiment = 'experiment name, default = %s' % d_experiment 
    h_detector   = 'detector name, default = %s' % d_detector
    h_ctype      = 'calibration constant type, default = %s' % d_ctype 
    h_dtype      = 'input file data type (None - array, txt - for text file, xtcav), default = %s' % d_dtype 
    h_run        = 'run number (begin), default = %s' % str(d_run) 
    h_run_end    = 'run number (end), default = %s' % str(d_run_end)
    h_time_stamp = 'time stamp, default = %s' % d_time_stamp 
    h_time_sec   = 'time (sec), default = %s' % str(d_time_sec)
    h_version    = 'version of constants, default = %s' % d_version
    h_confirm    = 'confirmation of the action, default = %s' % d_confirm
    h_iofname    = 'output file prefix, default = %s' % d_iofname
    h_comment    = 'comment to the document, default = %s' % d_comment
    h_loglevel   = 'logging level from list (%s), default = %s' % (LEVEL_NAMES, d_loglevel)

    parser = OptionParser(description='Command line interface to LCLS2 calibration data base', usage=usage())

    parser.add_option('--host',             default=d_host,       action='store', type='string', help=h_host)
    parser.add_option('--port',             default=d_port,       action='store', type='string', help=h_port)
    parser.add_option('-u', '--user',       default=d_user,       action='store', type='string', help=h_user)
    parser.add_option('-p', '--upwd',       default=d_upwd,       action='store', type='string', help=h_upwd)
    parser.add_option('--ctout',            default=d_ctout,      action='store', type='int',    help=h_ctout)
    parser.add_option('--stout',            default=d_stout,      action='store', type='int',    help=h_stout)
    parser.add_option('--dbname',           default=d_dbname,     action='store', type='string', help=h_dbname)
    parser.add_option('-d', '--detector',   default=d_detector,   action='store', type='string', help=h_detector)
    parser.add_option('-e', '--experiment', default=d_experiment, action='store', type='string', help=h_experiment)
    parser.add_option('-t', '--time_stamp', default=d_time_stamp, action='store', type='string', help=h_time_stamp)
    parser.add_option('-s', '--time_sec',   default=d_time_sec,   action='store', type='int',    help=h_time_sec)
    parser.add_option('-c', '--ctype',      default=d_ctype,      action='store', type='string', help=h_ctype)
    parser.add_option('-i', '--dtype',      default=d_dtype,      action='store', type='string', help=h_dtype)
    parser.add_option('-r', '--run',        default=d_run,        action='store', type='int',    help=h_run)
    parser.add_option('-R', '--run_end',    default=d_run_end,    action='store', type='string', help=h_run_end)
    parser.add_option('-v', '--version',    default=d_version,    action='store', type='string', help=h_version)
    parser.add_option('-C', '--confirm',    default=d_confirm,    action='store_true',           help=h_confirm)
    parser.add_option('-f', '--iofname',    default=d_iofname,    action='store', type='string', help=h_iofname)
    parser.add_option('-m', '--comment',    default=d_comment,    action='store', type='string', help=h_comment)
    parser.add_option('-l', '--loglevel',   default=d_loglevel,   action='store', type='string', help=h_loglevel)

    return parser
  
#------------------------------

if __name__ == "__main__" :
    cdb_cli()
    sys.exit(0)

#------------------------------
