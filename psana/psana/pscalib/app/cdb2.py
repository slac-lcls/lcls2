
"""
Created on 2018-02-23 by Mikhail Dubrovin
"""

import sys

import logging
logger = logging.getLogger(__name__)
from psana.pyalgos.generic.logger import config_logger, STR_LEVEL_NAMES
SCRNAME = sys.argv[0].rsplit('/')[-1]

import psana.pscalib.calib.CalibConstants as cc
from psana.pscalib.calib.MDBWeb_CLI_V2 import cdb_web, MODES  #  cdb  # includes import from MDB_CLI


USAGE = '\nCommand: cdb <mode> [options]'\
    '\n              modes: %s\n'%(', '.join(MODES))\
  + '\nExamples:\n'\
    '  cdb2\n'\
    '  cdb2 -h\n'\
    '  cdb2 print\n'\
    '  cdb2 print --dbname cdb_jungfrau_000003_mytestdb\n'\
    '  cdb2 print --dbname cdb_jungfrau_000003_mytestdb --colname jungfrau_000003\n'\
    '  cdb2 print --dbname cdb_jungfrau_000003_mytestdb --colname jungfrau_000003 --docid 688aa95d76d702c925246602\n'\
    '  cdb2 print --dbname cdb_testexper --colname testdet_1234 [--docid <document-id>]\n'\
    '  cdb2 print -d testdet_1234 [--docid <document-id>]\n'\
    '  cdb2 print -e testexper\n'\
    '  cdb2 print --dbname cdb_epixhr2x2_000001_mytestdb\n'\
    '  cdb2 add -k exp=mfx101332224,run=66 -d jungfrau -c pedestals -f clb-mfx101332224-jungfrau_000003-r0066-pedestals.data -S mytestdb -B 81 -R 85\n'\
    '  cdb2 get -k exp=mfx101332224,run=66 -d jungfrau -c pedestals -f myconsts -S mytestdb -L DEBUG\n'\
    '  cdb2 deldoc --dbname cdb_jungfrau_000003_mytestdb --colname jungfrau_000003 --docid 688aa95d76d702c925246602   -C\n'\
    '  cdb2 delcol --dbname cdb_jungfrau_000003_mytestdb --colname jungfrau_000003   -C\n'\
    '  cdb2 deldb  --dbname cdb_jungfrau_000003_mytestdb   -C\n'\
    '\n'\
    '  N/T cdb2 - stands for Not-Tested but should work\n'\
    '  N/T cdb2 add -k exp=cxic0415,run=100 -d cspad_0001 -c geometry -f mygeo -i txt\n'\
    '  N/T cdb2 add -k exp=amox27716,run=50 -d ele_opal -c pop_rbfs -f /sdf/group/lcls/ds/ana/detector/calib/misc/calib-amox27716-r50-opal-pop-rbfs-xiangli.json -i json\n'\
    '  N/T cdb2 add -k exp=amox27716,run=50 -d ele_opal -c pop_rbfs -f /sdf/group/lcls/ds/ana/detector/calib/misc/calib-amox27716-r50-opal-pop-rbfs-xiangli.pkl -i pkl\n'\
    '  N/T cdb2 add -k exp=exp12345,run=123 -d testdet_1234 -c test_ctype -f cm-confpars.txt -i txt -l DEBUG\n'\
    '  N/T cdb2 add -k exp=amox23616,run=104 -d xtcav -c pedestals -f xtcav_peds.data -i xtcav\n'\
    '\n'

#    '  N/T cdb2 add -k exp=exp12345,run=123 -d detector_1234 -c pedestals -f mypeds.data\n'\
#    '  N/T cdb2 add -k exp=new55555,run=123 -d detnew_5555   -c pedestals -f mypeds.data\n'\
#    '  N/T cdb2 add -k exp=amox27716,run=100 -d tmo_quadanode -c calibcfg -f configuration_quad.txt -i txt\n'\
#    '  N/T cdb2 add -k exp=rixx45619,run=1 -d epixhr2x2_000001 -c pedestals -f mypeds -S mytestdb # <== DEPLOY IN YOUR OWN DETECTOR-DB\n'\
#    '  N/T cdb2 add -k exp=rixx45619,run=1 -d epixhr2x2_000001 -t 2021-10-01T00:00:00-0800 -c pedestals -f mypeds -S mytestdb # <== DEPLOY IN YOUR OWN DETECTOR-DB\n'\
#    '  N/T cdb2 get -k exp=cxic0415,run= -e cxic0415 -d cspad_0001 -c pedestals -s 1520977960 -f mypeds  ??????\n'\
#    '  N/T cdb2 get -k exp=cxic0415,run=100 -d cspad_0001 -c geometry -f mygeo\n'\
#    '  N/T cdb2 deldb  --dbname cdb_testexper -C\n'\
#    '  N/T cdb2 delcol --dbname cdb_testexper --colname testdet_1234 -C\n'\
#    '  N/T cdb2 deldoc --dbname cdb_mfx101332224 --colname testdet_1234 --docid <document-id> -C\n'\
#    '\n'

#    '  cdb2 deldoc -e exp12345 -d detector_1234 -c pedestals -r 123 -v 05 -u <username> -p <password> -w -C\n'\
#    '  cdb2 deldoc -e cxix25615 -d cspad_0001 -c pedestals -r 125 -u <username> -p <password> -w -C\n'\
#    '  cdb2 deldoc -e cxix25615 -d cspad_0001 -c pedestals -s 1520977960 -u <username> -p <password> -w -C\n'\
#    '  cdb2 delcol -e cxix25615 -d cspad_0001 -u <username> -p <password> -w -C\n'\
#    '  cdb delcol -d cspad_0001 -u <username> -p <password> -w -C\n'\
#    '  cdb deldb -e amox23616 -u <username> -p <password> -w -C\n'\
#    '  cdb deldb -d opal1000_0059 -u <username> -p <password> -w -C\n'\
#    '  cdb delall\n'\
#    '   ??????cdb get -d cspad_0001 -c pedestals -r 100 -f mypeds\n'\
#    '   ??????cdb get -d opal1000_test -c pop_rbfs -r 50 -i pkl -f my_pop_rbfs\n'\
#    '  cdb add -k exp=,run= -e rixx45619 -d epixhr2x2_000001 -t 2021-10-01T00:00:00-0800 -c geometry -f mygeo -i txt -S mytestdb # <== DEPLOY IN YOUR OWN DETECTOR-DB\n'\
#    '  cdb get -k exp=,run= -e rixx45619 -d epixhr2x2_000001 -r 100 -c pedestals -f mypeds.txt -S mytestdb\n'\


def argument_parser():
    from argparse import ArgumentParser

    d_dskwargs   = None    # 'files=<fname.xtc>,exp=<expname>,run=<runs>,dir=<xtc-dir>, ...'
    d_mode       = 'print'
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
    d_run_beg    = None
    d_run_end    = 'end'
    d_time_stamp = None # '2001-09-08T18:46:40-0700'
    d_time_sec   = None
    d_version    = 'V2025-07-30'
    d_confirm    = False
    d_iofname    = None # './fname.txt'
    d_comment    = 'no comment'
    d_loglevel   = 'INFO'
    d_cdbonly    = True
    d_dbsuffix   = ''

    h_dskwargs= 'string of comma-separated (no spaces) simple parameters for DataSource(**kwargs),'\
                ' ex: exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
                ' or <fname.xtc> or files=<fname.xtc>'\
                ' or pythonic dict of generic kwargs, e.g.:'\
                ' \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dskwargs
    h_mode       = 'Mode of DB access, one of %s, default = %s' % (str(MODES), d_mode)
    h_ctout      = 'connect timeout connectTimeoutMS, default = %d' % d_ctout
    h_stout      = 'socket timeout serverSelectionTimeoutMS, default = %d' % d_stout
    h_dbname     = 'database name, works for mode "print" or "delete", default = %s' % d_dbname
    h_colname    = 'collection name, works for mode "print" or "delete", default = %s' % d_colname
    h_docid      = 'document Id, works for mode "print" or "delete", default = %s' % d_docid
    h_experiment = 'experiment name (for some commands, see examples), default = %s' % d_experiment
    h_detector   = 'detector name for run.Detector, default = %s' % d_detector
    h_ctype      = 'calibration constant type, default = %s' % d_ctype
    h_dtype      = 'i/o file data type (None - array, txt, xtcav, json, pkl), default = %s' % d_dtype
    h_run        = 'run number current/begin (for some commands, see examples), default = %s' % str(d_run)
    h_run_beg    = 'run number (begin if different from --run), default = %s' % str(d_run_beg)
    h_run_end    = 'run number (end), default = %s' % str(d_run_end)
    h_time_stamp = 'time stamp in format like 2020-05-22T01:02:03-0800, default = %s' % d_time_stamp
    h_time_sec   = 'time (sec), default = %s' % str(d_time_sec)
    h_version    = 'version of constants, default = %s' % d_version
    h_confirm    = 'confirmation of the action, default = %s' % d_confirm
    h_iofname    = 'output file prefix, default = %s' % d_iofname
    h_comment    = 'comment to the document, default = %s' % d_comment
    h_loglevel   = 'logging level from list (%s), default = %s' % (STR_LEVEL_NAMES, d_loglevel)
    h_cdbonly    = 'command valid for CDB only, ignores other DBs, default = %s' % d_cdbonly
    h_dbsuffix   = 'suffix of the PRIVATE DETECTOR-DB to deploy constants, default = %s' % str(d_dbsuffix)

    parser = ArgumentParser(description='CLI for LCLS2 calibration data base', usage=USAGE)

    parser.add_argument('-k', '--dskwargs',   default=d_dskwargs,   type=str, help=h_dskwargs)
    parser.add_argument('mode', nargs='?',    default=d_mode,       type=str, help=h_mode)
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
    parser.add_argument('-B', '--run_beg',    default=d_run_beg,    type=int, help=h_run_beg)
    parser.add_argument('-R', '--run_end',    default=d_run_end,    type=str, help=h_run_end)
    parser.add_argument('-v', '--version',    default=d_version,    type=str, help=h_version)
    parser.add_argument('-f', '--iofname',    default=d_iofname,    type=str, help=h_iofname)
    parser.add_argument('-m', '--comment',    default=d_comment,    type=str, help=h_comment)
    parser.add_argument('-L', '--loglevel',   default=d_loglevel,   type=str, help=h_loglevel)
    parser.add_argument('-S', '--dbsuffix',   default=d_dbsuffix,   type=str, help=h_dbsuffix)
    parser.add_argument('-C', '--confirm',    action='store_true',  help=h_confirm)
    parser.add_argument('--cdbonly',          action='store_false', help=h_cdbonly)

    return parser


def cdb_cli():
    """CLI for Calibration Data Base"""
    if len(sys.argv) < 2: sys.exit('\n%s\n\nEXIT DUE TO MISSING ARGUMENTS\n' % USAGE)

    parser = argument_parser()
    #args = parser.parse_args() # Namespace, e.g. args.detector
    #kwargs = vars(args)        # dict, e.g. kwargs['detector']
    cdb_web(parser)


if __name__ == "__main__":
    cdb_cli()
    sys.exit(0)

# EOF
