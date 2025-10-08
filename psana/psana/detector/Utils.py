"""
Utilities of common use for detector project
============================================

Usage::

  from psana.detector.Utils import *
  import psana.detector.Utils as ut

  is_selected = selected_record(nrec)
  s = info_dict(d, fmt='  %12s: %s', sep='\n')
  s = info_namespace(o, fmt='  %12s: %s', sep='\n')
  s = info_command_line(sep=' ')
  s = info_command_line_parameters(parser) # for OptionParser
  s = info_parser_arguments(parser) # for ArgumentParser
  save_log_record_at_start(dirrepo, procname, fac_mode=0o664, tsfmt='%Y-%m-%dT%H:%M:%S%z')
  resp = is_none(par, msg, logger_method=logger.debug)
  resp = is_true(cond, msg, logger_method=logger.debug)

2020-11-06 created by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

import os
import sys
import numpy as np

import psana.pyalgos.generic.Utils as gu
time, str_tstamp, get_login, get_hostname, get_cwd, save_textfile, load_textfile, get_enviroment,\
    set_file_access_mode, time_sec_from_stamp, create_directory, file_mode, change_file_ownership\
  = gu.time, gu.str_tstamp, gu.get_login, gu.get_hostname, gu.get_cwd, gu.save_textfile, gu.load_textfile, gu.get_enviroment,\
    gu.set_file_access_mode, gu.time_sec_from_stamp, gu.create_directory, gu.file_mode, gu.change_file_ownership

#log_rec_at_start = gu.log_rec_on_start

def selected_record(nrec):
    return nrec<5\
       or (nrec<50 and not nrec%10)\
       or (nrec<500 and not nrec%100)\
       or (not nrec%1000)


def inverse_dict(d):
    assert type(d) == dict
    return {v:k for k,v in d.items()}


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
       from optparse import ArgumentParser
       parser = ArgumentParser(...)
    """
    args = parser.parse_args()         # namespace of parameters
    kwas = vars(args)                  # dict of parameters
    defs = vars(parser.parse_args([])) # defaults only

    s = 'Command: ' + ' '.join(sys.argv)+\
        '\n  Argument list: %s\n  Optional parameters:\n' % str(args)+\
        '    <key>      <value>              <default>\n'
    for k,v in kwas.items():
        s += '    %s %s %s\n' % (k.ljust(10), str(v).ljust(20), str(defs[k]).ljust(20))
    return s


def info_parser_arguments(parser, title='optional parameters:'):
    """Prints input arguments and optional parameters
       from argparse import ArgumentParser
       parser = ArgumentParser(...)
    """
    args = parser.parse_args()
    opts = vars(args)
    defs = vars(parser.parse_args([])) # defaults only

    s = '%s\n' % title\
      + '    <key>      <value>              <default>\n'
    for k,v in opts.items():
        _v, _d = v, defs[k]
        if k in ('dirmode', 'filemode'):
           _v, _d = oct(_v), oct(_d)
        elif k == 'datbits':
           _v, _d = hex(_v), hex(_d)
        else:
           _v, _d = str(_v), str(_d)
        s += '    %s %s %s\n' % (k.ljust(10), _v.ljust(20), _d.ljust(20))
    return s

info_command_line_arguments = info_parser_arguments


def log_rec_at_start(tsfmt='%Y-%m-%dT%H:%M:%S%z', **kwa):
    """DEPRECATED - moved to detector.RepoManager.py
       Returns (str) record containing timestamp, login, host, cwd, and command line
    """
    s_kwa = ' '.join(['%s:%s'%(k,str(v)) for k,v in kwa.items()])
    return '\n%s user:%s@%s cwd:%s %s command:%s'%\
           (str_tstamp(fmt=tsfmt), get_login(), get_hostname(), get_cwd(), s_kwa, ' '.join(sys.argv))


def save_log_record_at_start(dirrepo, procname, dirmode=0o2775, filemode=0o664, logmode='INFO', group='ps-users',\
                             tsfmt='%Y-%m-%dT%H:%M:%S%z', umask=0o0):
    """Adds record at start to the log file <dirrepo>/logs/log-<procname>-<year>.txt."""
    from psana.detector.RepoManager import RepoManager
    os.umask(umask)
    rec = log_rec_at_start(tsfmt, **{'dirrepo':dirrepo,})
    logger.debug('Record at start: %s' % rec)
    repoman = RepoManager(dirrepo=dirrepo, dirmode=dirmode, filemode=filemode, umask=umask, group=group)
    logatstart = repoman.logname_at_start(procname)
    fexists = os.path.exists(logatstart)
    save_textfile(rec, logatstart, mode='a')
    if not fexists: set_file_access_mode(logatstart, filemode)
    logger.info('Record: %s\nSaved: %s' % (rec, logatstart))
    #--
    #logfname = repoman.logname('%s_%s' % (procname, get_login()))
    #repoman.save_record_at_start(procname, tsfmt=tsfmt, adddict={'logfile':logfname})
    #return repoman


def save_record_at_start(repoman, procname, tsfmt='%Y-%m-%dT%H:%M:%S%z', adddict={}):
    """DEPRECATED - moved to detector.RepoManager.py"""
    os.umask(repoman.umask)

    logatstart = repoman.logname_at_start(procname)
    fexists = os.path.exists(logatstart)
    d = {'dirrepo':repoman.dirrepo, 'logfile':repoman.logname(procname)}
    if adddict: d.update(adddict)
    rec = log_rec_at_start(tsfmt, **d)
    save_textfile(rec, logatstart, mode='a')
    if not fexists:
        set_file_access_mode(logatstart, repoman.filemode)
        change_file_ownership(logatstart, user=None, group=repoman.group)
    logger.info('Record: %s\nSaved: %s' % (rec, logatstart))


def is_none(par, msg, logger_method=logger.debug):
    resp = par is None
    if resp: logger_method(msg)
    return resp


def is_true(cond, msg, logger_method=logger.debug):
    if cond: logger_method(msg)
    return cond

# EOF
