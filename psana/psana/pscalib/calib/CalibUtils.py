#------------------------------
"""
:py:class:`CalibUtils` - set of utilities for calibration algorithms
==========================================================================

Usage ::

    #Test this module:  python lcls2/psana/psana/pscalib/calib/CalibUtils.py
    from psana.pscalib.calib.CalibUtils import *

    # see test

 See:
    - :py:class:`CalibUtils`
    - `matplotlib <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2018-03-15 by Mikhail Dubrovin
"""
#------------------------------

import os
import psana.pyalgos.generic.Utils as gu

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
    if listdicts is None : return None
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

def info_history_dict(d, fmt='  %s : %s\n', cmt='HISTORY info:') :
    if d is None : return
    s = '%s\n' % cmt
    for k, v in d.items() :
        s += fmt % (k.ljust(10),v)
    return s

#------------------------------

def print_history_dict(d, fmt='    %s : %s\n', cmt='HISTORY info:') :
    print(info_history_dict(d, fmt, cmt))

#------------------------------

if __name__ == "__main__" :

  def test_history_access() :
    #from psana.pscalib.calib.CalibUtils import *
    hfname = '/reg/d/psdm/xpp/xpptut15/calib/CsPad::CalibV1/XppGon.0:Cspad.0/pedestals/HISTORY'
    listdicts = history_list_of_dicts(hfname, verb=True)
    d = history_dict_for_file(listdicts, '54-end.data')
    print_history_dict(d)

#------------------------------

if __name__ == "__main__" :
    import sys
    test_history_access()
    sys.exit('End of test %s')

#------------------------------

