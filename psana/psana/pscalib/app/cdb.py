
"""
Created on 2018-02-23 by Mikhail Dubrovin
"""

import sys

import logging
logger = logging.getLogger(__name__)
from psana.pyalgos.generic.logger import config_logger, STR_LEVEL_NAMES
SCRNAME = sys.argv[0].rsplit('/')[-1]

import psana.pscalib.calib.CalibConstants as cc
from psana.pscalib.calib.MDBWeb_CLI import cdb_web, cdb, MODES # includes import from MDB_CLI


USAGE = '\nCommand: cdb <mode> [options]'\
    '\n              modes: %s\n'%(', '.join(MODES))\
  + '\nExamples:\n'\
    '  cdb\n'\
    '  cdb -h\n'\
    '  cdb print\n'\
    '  cdb print --dbname cdb_testexper --colname testdet_1234 [--docid <document-id>]\n'\
    '  cdb print -d testdet_1234 [--docid <document-id>]\n'\
    '  cdb print -e testexper\n'\
    '  cdb convert -e cxif5315 -p <password>\n'\
    '  cdb convert -e amox23616 -p <password>\n'\
    '  cdb get -e testexper -d testdet_1234 -r 21 -c test_ctype\n'\
    '  cdb get -e exp12345 -d detector_1234 -c testdict -r 23 -f mydict\n'\
    '  cdb get -e cxic0415 -d cspad_0001 -c pedestals -s 1520977960 -f mypeds\n'\
    '  cdb get -e cxic0415 -d cspad_0001 -c geometry -r 100 -f mygeo\n'\
    '  cdb get -d cspad_0001 -c pedestals -r 100 -f mypeds\n'\
    '  cdb get -d opal1000_test -c pop_rbfs -r 50 -i pkl -f my_pop_rbfs\n'\
    '  cdb add -e cxic0415 -d cspad_0001 -c geometry -r 100 -f mygeo -i txt -l DEBUG\n'\
    '  cdb add -e testexper -d testdet_1234 -c test_ctype -r 123 -f cm-confpars.txt -i txt -l DEBUG\n'\
    '  cdb add -e exp12345 -d detector_1234 -c pedestals -r 123 -f mypeds.data\n'\
    '  cdb add -e new55555 -d detnew_5555   -c pedestals -r 123 -f mypeds.data\n'\
    '  cdb add -e amox27716 -d tmo_quadanode -c calibcfg -r 100 -f configuration_quad.txt -i txt\n'\
    '  cdb add -e amox27716 -d ele_opal -c pop_rbfs -r 50 -f /reg/g/psdm/detector/calib/misc/calib-amox27716-r50-opal-pop-rbfs-xiangli.json -i json\n'\
    '  cdb add -e amox27716 -d ele_opal -c pop_rbfs -r 50 -f /reg/g/psdm/detector/calib/misc/calib-amox27716-r50-opal-pop-rbfs-xiangli.pkl -i pkl\n'\
    '  cdb add -e amox23616 -d xtcav -c pedestals -r 104 -f xtcav_peds.data -i xtcav\n'\
    '  cdb add -e rixx45619 -d epixhr2x2_000001 -r1 -c pedestals -f mypeds -S mytestdb # <== DEPLOY IN YOUR OWN DETECTOR-DB\n'\
    '  cdb add -e rixx45619 -d epixhr2x2_000001 -t 2021-10-01T00:00:00-0800 -c pedestals -f mypeds -S mytestdb # <== DEPLOY IN YOUR OWN DETECTOR-DB\n'\
    '  cdb add -e rixx45619 -d epixhr2x2_000001 -t 2021-10-01T00:00:00-0800 -c geometry -f mygeo -i txt -S mytestdb # <== DEPLOY IN YOUR OWN DETECTOR-DB\n'\
    '  cdb get -e rixx45619 -d epixhr2x2_000001 -r 100 -c pedestals -f mypeds.txt -S mytestdb\n'\
    '  cdb print --dbname cdb_epixhr2x2_000001_mytestdb\n'\
    '  cdb deldb  --dbname cdb_testexper -C\n'\
    '  cdb delcol --dbname cdb_testexper --colname testdet_1234 -C\n'\
    '  cdb deldoc --dbname cdb_testexper --colname testdet_1234 --docid <document-id> -C\n'\
    '  cdb deldoc -e exp12345 -d detector_1234 -c pedestals -r 123 -v 05 -u <username> -p <password> -w -C\n'\
    '  cdb deldoc -e cxix25615 -d cspad_0001 -c pedestals -r 125 -u <username> -p <password> -w -C\n'\
    '  cdb deldoc -e cxix25615 -d cspad_0001 -c pedestals -s 1520977960 -u <username> -p <password> -w -C\n'\
    '  cdb delcol -e cxix25615 -d cspad_0001 -u <username> -p <password> -w -C\n'\
    '  cdb delcol -d cspad_0001 -u <username> -p <password> -w -C\n'\
    '  cdb deldb -e amox23616 -u <username> -p <password> -w -C\n'\
    '  cdb deldb -d opal1000_0059 -u <username> -p <password> -w -C\n'\
    '  cdb delall\n'\
    '  cdb export --dbname cdb_exp12345\n'\
    '  cdb import --dbname cdb_exp12345 --iofname cdb-...arc\n'\
    '  cdb print --host=psanagpu115 --port=27017 --stout=1000'


def argument_parser():
    from argparse import ArgumentParser

    d_mode       = 'print'
    d_host       = cc.HOST
    d_port       = cc.PORT
    d_user       = cc.USERNAME
    d_upwd       = ''
    d_ctout      = 5000
    d_stout      = 30000
    d_dbname     = None
    d_colname    = None
    d_docid      = None
    d_experiment = None
    d_detector   = None
    d_ctype      = None # cc.list_calib_names[0], 'pedestals'
    d_dtype      = None
    d_run        = 0
    d_run_end    = 'end'
    d_time_stamp = None # '2001-09-08T18:46:40-0700'
    d_time_sec   = None
    d_version    = None # 'V2021-10-08'
    d_confirm    = False
    d_iofname    = None # './fname.txt'
    d_comment    = 'no comment'
    d_loglevel   = 'INFO'
    d_webcli     = True
    d_cdbonly    = True
    d_dbsuffix   = ''

    h_mode       = 'Mode of DB access, one of %s, default = %s' % (str(MODES), d_mode)
    h_host       = 'DB host, default = %s' % d_host
    h_port       = 'DB port, default = %s' % d_port
    h_user       = 'username to access DB, default = %s' % d_user
    h_upwd       = 'password, default = %s' % d_upwd
    h_ctout      = 'connect timeout connectTimeoutMS, default = %d' % d_ctout
    h_stout      = 'socket timeout serverSelectionTimeoutMS, default = %d' % d_stout
    h_dbname     = 'database name, works for mode "print" or "delete", default = %s' % d_dbname
    h_colname    = 'collection name, works for mode "print" or "delete", default = %s' % d_colname
    h_docid      = 'document Id, works for mode "print" or "delete", default = %s' % d_docid
    h_experiment = 'experiment name, default = %s' % d_experiment
    h_detector   = 'detector name, default = %s' % d_detector
    h_ctype      = 'calibration constant type, default = %s' % d_ctype
    h_dtype      = 'i/o file data type (None - array, txt, xtcav, json, pkl), default = %s' % d_dtype
    h_run        = 'run number (begin), default = %s' % str(d_run)
    h_run_end    = 'run number (end), default = %s' % str(d_run_end)
    h_time_stamp = 'time stamp in format like 2020-05-22T01:02:03-0800, default = %s' % d_time_stamp
    h_time_sec   = 'time (sec), default = %s' % str(d_time_sec)
    h_version    = 'version of constants, default = %s' % d_version
    h_confirm    = 'confirmation of the action, default = %s' % d_confirm
    h_iofname    = 'output file prefix, default = %s' % d_iofname
    h_comment    = 'comment to the document, default = %s' % d_comment
    h_loglevel   = 'logging level from list (%s), default = %s' % (STR_LEVEL_NAMES, d_loglevel)
    h_webcli     = 'use web-based CLI, default = %s' % d_webcli
    h_cdbonly    = 'command valid for CDB only, ignores other DBs, default = %s' % d_cdbonly
    h_dbsuffix   = 'suffix of the PRIVATE DETECTOR-DB to deploy constants, default = %s' % str(d_dbsuffix)

    parser = ArgumentParser(description='CLI for LCLS2 calibration data base', usage=USAGE)

    parser.add_argument('mode', nargs='?',    default=d_mode,       type=str, help=h_mode)
    parser.add_argument('--host',             default=d_host,       type=str, help=h_host)
    parser.add_argument('--port',             default=d_port,       type=str, help=h_port)
    parser.add_argument('-u', '--user',       default=d_user,       type=str, help=h_user)
    parser.add_argument('-p', '--upwd',       default=d_upwd,       type=str, help=h_upwd)
    parser.add_argument('--ctout',            default=d_ctout,      type=int, help=h_ctout)
    parser.add_argument('--stout',            default=d_stout,      type=int, help=h_stout)
    parser.add_argument('--dbname',           default=d_dbname,     type=str, help=h_dbname)
    parser.add_argument('--colname',          default=d_colname,    type=str, help=h_colname)
    parser.add_argument('--docid',            default=d_docid,      type=str, help=h_docid)
    parser.add_argument('-d', '--detector',   default=d_detector,   type=str, help=h_detector)
    parser.add_argument('-e', '--experiment', default=d_experiment, type=str, help=h_experiment)
    parser.add_argument('-t', '--time_stamp', default=d_time_stamp, type=str, help=h_time_stamp)
    parser.add_argument('-s', '--time_sec',   default=d_time_sec,   type=int, help=h_time_sec)
    parser.add_argument('-c', '--ctype',      default=d_ctype,      type=str, help=h_ctype)
    parser.add_argument('-i', '--dtype',      default=d_dtype,      type=str, help=h_dtype)
    parser.add_argument('-r', '--run',        default=d_run,        type=int, help=h_run)
    parser.add_argument('-R', '--run_end',    default=d_run_end,    type=str, help=h_run_end)
    parser.add_argument('-v', '--version',    default=d_version,    type=str, help=h_version)
    parser.add_argument('-f', '--iofname',    default=d_iofname,    type=str, help=h_iofname)
    parser.add_argument('-m', '--comment',    default=d_comment,    type=str, help=h_comment)
    parser.add_argument('-l', '--loglevel',   default=d_loglevel,   type=str, help=h_loglevel)
    parser.add_argument('-S', '--dbsuffix',   default=d_dbsuffix,   type=str, help=h_dbsuffix)
    parser.add_argument('-C', '--confirm',    action='store_true',  help=h_confirm)
    parser.add_argument('-w', '--webcli',     action='store_false', help=h_webcli)
    parser.add_argument('--cdbonly',          action='store_false', help=h_cdbonly)

    return parser


def cdb_cli():
    """CLI for Calibration Data Base"""
    if len(sys.argv) < 2: sys.exit('\n%s\n\nEXIT DUE TO MISSING ARGUMENTS\n' % USAGE)

    parser = argument_parser()
    args = parser.parse_args()  # Namespace
    #kwargs = vars(args)        # dict

    if args.webcli: cdb_web(parser)
    else:           cdb(parser)


if __name__ == "__main__":
    cdb_cli()
    sys.exit(0)

# EOF
