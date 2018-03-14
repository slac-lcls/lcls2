
#------------------------------
"""
:py:class:`CalibUtils` - generic utilities for Calib project
===================================================================

Usage ::

    # python lcls2/psana/pscalib/calib/CalibUtils.py

    from psana.pscalib.calib.CalibUtils import *

See:
 * :py:class:`CalibBase`
 * :py:class:`CalibConstant`

For more detail see `Calibration Store <https://confluence.slac.stanford.edu/display/PCDS/MongoDB+evaluation+for+calibration+store>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2018-03-05 by Mikhail Dubrovin
"""
#------------------------------
import os
#import numpy as np

#from psana.pyalgos.generic.PSConstants import INSTRUMENTS, DIR_INS, DIR_FFB # , DIR_LOG
from psana.pyalgos.generic.PSNameManager import nm
import psana.pyalgos.generic.Utils as gu
import psana.pscalib.calib.CalibConstants as cc # list_calib_names

from psana.pscalib.calib.NDArrIO import load_txt
import psana.pyalgos.generic.NDArrUtils as ndu

import psana.pscalib.calib.MDBUtils as dbu # insert_constants, time_and_timestamp

#------------------------------

def history_list_of_dicts(history, verb=True) :
    """Returns list of dictionaries from history file.
    """
    if not os.path.exists(history) : return None
    recs = gu.load_textfile(history, verb).split('\n')        
    return [dict([f.split(':',1) for f in r.split()]) for r in recs if r]

#------------------------------

def history_dict_for_file(listdicts, fname) :
    """Returns dictionary-record for from history_list_of_dicts.
    """
    for d in listdicts :
        v = d.get('file', None)
        if v == None : continue
        if v == fname : return d
    return None

#------------------------------

def parse_calib_file_name(fname) :
    """Returns splitted parts of the calibration file name,
       e.g: '123-end.data' -> ('123', 'end', '.data')
    """
    fnfields = os.path.splitext(fname)      # '123-end.data' -> ('123-end', 'data') 
    if len(fnfields) != 2 : return None     # check that file name like '123-end.data' has splited extension
    if fnfields[1] != '.data': return None  # check extension
    fnparts = fnfields[0].split('-')        # '123-end' -> ('123', 'end')
    if len(fnparts) != 2 : return None      # check that file name has two parts

    #print('XXX parts of the file name:', fnparts[0], fnparts[1], fnfields[1])
    return fnparts[0], fnparts[1], fnfields[1]

#------------------------------

def print_history_dict(d) :
    if d is None : return
    print('HISTORY as dict:')
    for k, v in d.items() :
        print('%s%s : %s' % (7*' ',k,v))

#------------------------------

def add_calib_file_to_cdb(exp, dircalib, calibvers, detname, cftype, fname, cfdir, listdicts, **kwargs) :
    """
    """
    verbose = kwargs.get('verbose', False)

    d = history_dict_for_file(listdicts, fname)
    #if verbose : print_history_dict(d)

    resp = parse_calib_file_name(fname)
    begin, end, ext = resp if resp is not None else (None, None, None)
    fpath = '%s/%s' % (cfdir, fname)
    data = gu.load_textfile(fpath, verbose) if cftype == 'geometry' else\
           load_txt(fpath) # using NDArrIO

    if verbose :
        ndu.print_ndarr(data, 'scan calib: data')
        print('scan calib:', exp, cfdir, fname, begin, end, ext, calibvers, detname, cftype)

    kwargs['run']        = begin
    kwargs['run_end']    = end
    kwargs['detector']   = detname
    kwargs['ctype']      = cftype
    kwargs['time_sec']   = '1000000000'
    kwargs['time_stamp'] = None
    kwargs['extpars']    = d # just in case save entire history dict
    #kwargs['comment']    = 'HISTORY: %s' % d.get('comment', '')

    dbu.insert_calib_data(data, **kwargs)
    #dbu.insert_constants(data, d['experiment'], d['detector'], d['ctype'], d['run'], d['time_sec'], d['version'], **kwargs)

#------------------------------

def scan_calib_for_experiment(exp='cxix25615', **kwargs) :

    host    = kwargs.get('host', None)
    port    = kwargs.get('port', None)
    verbose = kwargs.get('verbose', False)

    client = dbu.connect_to_server(host, port)
    dbname = dbu.db_prefixed_name(exp)
    if dbu.database_exists(client, dbname) :
        print('Experiment %s already has a database. Consider to delete it from the list:\n%s'%\
              (exp, str(dbu.database_names(client))))
        print('Use command: cdb deletedb --dbname %s' % dbname)
        return

    dircalib = nm.dir_calib(exp)
    #if verbose : 
    print('Scan: %s' % dircalib)

    for dir0 in gu.get_list_of_files_in_dir_for_part_fname(dircalib, pattern='::') :
        if not os.path.isdir(dir0) : continue
        calibvers = os.path.basename(dir0)
        if verbose : print('  %s ' % calibvers)

        for dir1 in gu.get_list_of_files_in_dir_for_part_fname(dir0, pattern=':') :
            if not os.path.isdir(dir1) : continue
            detname = os.path.basename(dir1)
            detname_m = detname.replace(":","-").replace(".","-").lower()
            if verbose : print('    %s' % detname_m)        

            for cftype in gu.get_list_of_files_in_dir(dir1) :
                if not(cftype in cc.list_calib_names) : continue
                dir2 = '%s/%s' % (dir1, cftype)
                if not os.path.isdir(dir2) : continue
                if verbose : print('      %s' % cftype)

                cfdir = '%s/%s/%s/%s' % (dircalib, calibvers, detname, cftype)
                listdicts = history_list_of_dicts('%s/HISTORY' % cfdir, verbose)
                #if verbose : print('XXX listdicts', listdicts)
                count = 0
                for fname in gu.get_list_of_files_in_dir(dir2) :
                    if verbose : print('        %s' % fname)
                    if fname == 'HISTORY' : continue
                    add_calib_file_to_cdb(exp, dircalib, calibvers, detname_m, cftype, fname, cfdir, listdicts, **kwargs)
                    count += 1

                print('  converted %3d files from: %s' % (count, cfdir))


#------------------------------

if __name__ == "__main__" :
  def usage() : return 'Use command: python %s <test-number>, where <test-number> = 1,2,...,8,...' % sys.argv[0]

#------------------------------

  def test_dir_calib(tname) :
      print('dir_calib: %s' % nm.dir_calib('cxi02117'))

#------------------------------

  def test_all(tname) :
    print('\n%s\n' % usage())
    if len(sys.argv) != 2 : test_dir_calib(tname)
    elif tname == '1': test_dir_calib(tname)
    elif tname == '2': scan_calib_for_experiment('cxix25615')
    else : sys.exit('Test number parameter is not recognized.\n%s' % usage())

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test_all(tname)
    sys.exit('End of Test %s' % tname)

#------------------------------
