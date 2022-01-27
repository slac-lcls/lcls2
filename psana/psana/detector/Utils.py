"""
Utilities of common use for detector project
============================================

Usage::

  from psana.detector.Utils *

  is_selected = selected_record(nrec)
  s = info_dict(d, fmt='  %12s: %s', sep='\n')
  s = info_namespace(o, fmt='  %12s: %s', sep='\n')
  s = info_command_line(sep=' ')
  s = info_command_line_parameters(parser) # for OptionParser
  s = info_parser_arguments(parser) # for ArgumentParser
  save_log_record_on_start(dirrepo, procname, fac_mode=0o777, tsfmt='%Y-%m-%dT%H:%M:%S%z')

2020-11-06 created by Mikhail Dubrovin
"""

#import logging
#logger = logging.getLogger(__name__)

import sys
import numpy as np

def selected_record(nrec):
    return nrec<5\
       or (nrec<50 and not nrec%10)\
       or (nrec<500 and not nrec%100)\
       or (not nrec%1000)


def info_dict(d, fmt='  %12s: %s', sep='\n', sepnext=13*' '):
    return (sep if sep[0]!=',' else '')\
         + sep.join([fmt % (k, info_dict(v, fmt=fmt, sep=sep+sepnext)\
               if isinstance(v,dict) else str(v)) for k,v in d.items()])


def info_namespace(o, fmt='  %12s: %s', sep='\n'):
    return sep.join([fmt %(n,str(getattr(o,n,None))) for n in dir(o) if n[0]!='_'])


def info_command_line(sep=' '):
    return sep.join(sys.argv)


def info_command_line_parameters(parser):
    """Prints input arguments and optional parameters
       from optparse import OptionParser
       parser = OptionParser(...)
    """
    (popts, pargs) = parser.parse_args()
    args = pargs                             # list of positional arguments
    opts = vars(popts)                       # dict of options
    defs = vars(parser.get_default_values()) # dict of default options

    s = 'Command: ' + ' '.join(sys.argv)+\
        '\n  Argument list: %s\n  Optional parameters:\n' % str(args)+\
        '    <key>      <value>              <default>\n'
    for k,v in opts.items():
        s += '    %s %s %s\n' % (k.ljust(10), str(v).ljust(20), str(defs[k]).ljust(20))
    return s


def info_parser_arguments(parser):
    """Prints input arguments and optional parameters
       from argparse import ArgumentParser
       parser = ArgumentParser(...)
    """
    args = parser.parse_args()
    opts = vars(args)
    defs = vars(parser.parse_args([])) # defaults only

    s = 'Optional parameters:\n'\
        '    <key>      <value>              <default>\n'
    for k,v in opts.items():
        s += '    %s %s %s\n' % (k.ljust(10), str(v).ljust(20), str(defs[k]).ljust(20))
    return s

info_command_line_arguments = info_parser_arguments


def save_log_record_on_start(dirrepo, procname, dirmode=0o777, filemode=0o666, tsfmt='%Y-%m-%dT%H:%M:%S%z'):
    """Adds record on start to the log file <dirrepo>/logs/log-<procname>-<year>.txt
    """
    from psana.pyalgos.generic.Utils import os, logger, log_rec_on_start, str_tstamp, create_directory, save_textfile, set_file_access_mode

    rec = log_rec_on_start(tsfmt)
    logger.debug('Record on start: %s' % rec)
    year = str_tstamp(fmt='%Y')
    create_directory(dirrepo, dirmode)
    dirlog = '%s/logs' % dirrepo
    create_directory(dirlog, dirmode)
    logfname = '%s/%s_log_%s.txt' % (dirlog, year, procname)
    fexists = os.path.exists(logfname)
    save_textfile(rec, logfname, mode='a')
    if not fexists: set_file_access_mode(logfname, filemode)
    logger.info('Saved: %s' % logfname)
# EOF
