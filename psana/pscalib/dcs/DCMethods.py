#!/usr/bin/env python
#------------------------------
"""
:py:class:`DCMethods` is a set of utilities for direct operations with calibration data
=======================================================================================

Usage::

    # Import
    import psana
    import PSCalib.DCMethods as dcm

    # Example of parameters
    dsname = 'exp=cxif5315:run=129'
    # or:
    dsname = '/reg/g/psdm/detector/data_test/xtc/cxif5315-e545-r0169-s00-c00.xtc'
    ds = psana.DataSource(dsname)
    env=ds.env()
    evt=ds.events().next()

    src      = 'Cspad.' # 'Epix100a.', etc
    ctype    = gu.PIXEL_MASK # | gu.PEDESTALS | gu.PIXEL_STATUS, etc.
    vers     = None # or e.g. 5
    calibdir = None # or e.g. './calib'
    nda      = np.zeros((32,185,388))
    pred     = 'CxiDs2.0:Cspad.0'
    succ     = 'CxiDs2.0:Cspad.0'
    range    = '1474587520-end'

    par      = evt # psana.Event | float - tsec event time
    parts    = env # psana.Env | psana.Event | float - tsec event time

    # Methods with dynamically-reconstructed calib file name
    dcm.add_constants(nda, par, env, src='Epix100a.', ctype=gu.PIXEL_MASK, calibdir=None, vers=None, pred=None, succ=None, cmt=None, verb=False)
    dcm.print_content(env, src='Epix100a.', calibdir=None)
    nda = dcm.get_constants(par, env, src='Epix100a.', ctype=gu.PIXEL_MASK, calibdir=None, vers=None, verb=False)
    dcm.delete_version(evt, env, src='Epix100a.', ctype=gu.PIXEL_MASK, calibdir=None, vers=None, cmt=None, verb=False)
    dcm.delete_range  (evt, env, src='Epix100a.', ctype=gu.PIXEL_MASK, calibdir=None, range=None, cmt=None, verb=False)
    dcm.delete_ctype  (evt, env, src='Epix100a.', ctype=gu.PIXEL_MASK, calibdir=None, cmt=None, verb=False)

    # Methods using fname
    dcm.add_constants_to_file(data, fname, parts, env, ctype=gu.PIXEL_MASK, vers=None, pred=None, succ=None, cmt=None, verb=False)
    dcm.print_content_from_file(fname)
    nda = dcm.get_constants_from_file(fname, parts, ctype=gu.PIXEL_MASK, vers=None, verb=False)
    dcm.delete_version_from_file(fname, parts, ctype=gu.PIXEL_MASK, vers=None, cmt=None, verb=False)
    dcm.delete_range_from_file  (fname, ctype=gu.PIXEL_MASK, range=None, cmt=None, verb=False)
    dcm.delete_ctype_from_file  (fname, ctype=gu.PIXEL_MASK, cmt=None, verb=False)

Methods 
    * :meth:`add_constants`, 
    * :meth:`add_constants_to_file`, 
    * :meth:`print_content`, 
    * :meth:`print_content_from_file`, 
    * :meth:`get_constants`, 
    * :meth:`get_constants_from_file`, 
    * :meth:`delete_version`, 
    * :meth:`delete_version_from_file`, 
    * :meth:`delete_range`, 
    * :meth:`delete_range_from_file`, 
    * :meth:`delete_ctype`
    * :meth:`delete_ctype_from_file`
 
Classes:
    * :class:`DCStore`
    * :class:`DCType`
    * :class:`DCRange`
    * :class:`DCVersion`
    * :class:`DCBase`
    * :class:`DCInterface`
    * :class:`DCUtils`
    * :class:`DCDetectorId`
    * :class:`DCConfigParameters`
    * :class:`DCFileName`
    * :class:`DCLogger`
    * :class:`DCMethods`
    * :class:`DCEmail`

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created: 2016-09-23 by Mikhail Dubrovin
"""
#------------------------------

import sys
import os
#import numpy as np
#import h5py
from time import time #localtime, strftime

from PSCalib.DCLogger import log
from PSCalib.DCFileName import DCFileName
import PSCalib.DCUtils as dcu
from PSCalib.DCStore import DCStore

sp = dcu.sp
gu = dcu.gu

#------------------------------

def is_good_fname(fname, verb=False) :
    """Checks the hdf5 file name validity, returns True or False.
    
    Parameters
    
    - fname : str - full path to the file
    - verb : bool - verbosity, default=False - do not print any message

    Returns

    True/False - for existing or not file
    """
    metname = sys._getframe().f_code.co_name

    if fname is None :
        if verb : print '%s WARNING: file name is None' % metname
        return False

    if not isinstance(fname, str) :
        if verb : print '%s WARNING: parameter fname is not str' % metname
        return False

    if not os.path.exists(fname) :
        if verb : print '%s WARNING: file %s does not exist' % (metname, fname)
        return False

    return True

#------------------------------
#------------------------------
#------------------------------
#------------------------------

def add_constants_to_file(data, fname, par, env=None, ctype=gu.PIXEL_MASK,\
                          vers=None,\
                          pred=None,\
                          succ=None,\
                          cmt=None,\
                          verb=False) :
    """Adds specified numpy array to the hdf5 file.

    Parameters
    
    - data : numpy.array or str - array or string of calibration constants/data to save in file
    - fname: full path to the hdf5 file
    - par  : psana.Event | psana.Env | float - tsec event time
    - env  : psana.Env -> is used to get exp=env.experiment() for comments etc.
    - ctype: gu.CTYPE - enumerated calibration type, e.g.: gu.PIXEL_MASK
    - vers : int - calibration version
    - pred : str - predecessor name
    - succ : str - successor name
    - cmt  : str - comment saved as a history record within DCRange
    - verb : bool - verbosity, default=False - do not print any message

    See :py:class:`DCMethods`
    """
    metname = sys._getframe().f_code.co_name

    str_ctype = gu.dic_calib_type_to_name[ctype]

    if verb : print '  %s.add_constants_to_file  ctype: %s  vers: %s'%\
                    (metname, str_ctype, vers)

    if fname is None :
        if verb : print 'WARNING: file name is not defined - return None'
        return

    tsec_ev = dcu.par_to_tsec(par)

    cs = DCStore(fname)

    if verb : log.setPrintBits(0377) # 0377

    if os.path.exists(fname) : cs.load()

    cs.set_tscfile(tsec=tsec_ev)
    cs.set_predecessor(pred)
    cs.set_successor(succ)

    msg = 'detname:%s predecessor:%s successor:%s ts:%.0f' %\
          (cs.detname(), cs.predecessor(), cs.successor(), cs.tscfile())
    #cs.add_history_record('%s for %s' % (metname, msg))
    #cs.add_par('par-1-in-DCStore', 1)

    ct = cs.add_ctype(str_ctype, cmt='')
    if ct is None : return
    #ct.add_history_record('%s - add DCType %s' % (metname,str_ctype))
    #ct.add_par('par-1-in-DCType', 1)

    exp = env.experiment() if env is not None else ''
    exp = exp if exp != '' else 'unknown'
    runnum = par.run() if not isinstance(par,float) else 0 
    msg = 'exp=%s:run=%s' % (exp, str(runnum))
    cr = ct.add_range(tsec_ev, end=None, cmt=msg)
    if cr is None : return
    cr.add_par('experiment', exp)
    cr.add_par('run', str(runnum))
    #cr.set_vnum_def(vnum=None)

    msg = '' if cmt is None else cmt
    cv = cr.add_version(vnum=vers, tsec_prod=time(), nda=data, cmt=msg)
    if cv is None : return
    #v = cr.vnum_last() if vers is None else vers
    #rec='%s vers=%d: %s' % (metname, v, cmt if cmt is not None else 'no-comments') 
    #cr.add_history_record(rec)

    if verb : 
        print 50*'_','\nIn %s:' % metname
        cs.print_obj()
    
    if verb : log.setPrintBits(02) # 0377

    cs.save()

#------------------------------

def add_constants(data, par, env, src='Epix100a.', ctype=gu.PIXEL_MASK, calibdir=None,\
                  vers=None,\
                  pred=None,\
                  succ=None,\
                  cmt=None,\
                  verb=False) :
    """Adds specified numpy array to the hdf5 file.

    Parameters
    
    - data : numpy.array or str - array or string of calibration constants/data to save in file
    - env  : psana.Env -> full detector name for psana.Source -> hdf5 file name
    - par  : psana.Event | float time -> event time
    - src  : str - source short/full name, alias or full -> hdf5 file name
    - ctype: gu.CTYPE - enumerated calibration type, e.g.: gu.PIXEL_MASK
    - calibdir : str - fallback path to calib dir (if xtc file is copied - calib and experiment name are lost)
    - vers : int - calibration version
    - pred : str - predecessor name
    - succ : str - successor name
    - cmt  : str - comment saved as a history record within DCRange
    - verb : bool - verbosity, default=False - do not prnt any message

    See :py:class:`DCMethods`
    """
    metname = sys._getframe().f_code.co_name

    str_ctype = gu.dic_calib_type_to_name[ctype]

    if verb : print '  %s.add_constants  src: %s\n  ctype: %s\n  vers: %s\n  calibdir:%s'%\
                    (metname, src, str_ctype, vers, calibdir)

    ofn = DCFileName(env, src, calibdir)
    if verb : ofn.print_attrs()
    ofn.make_path_to_calib_file() # depth=2, mode=0775)

    fname = ofn.calib_file_path()

    add_constants_to_file(data, fname, par, env, ctype, vers, pred, succ, cmt, verb)

#------------------------------

def get_constants_from_file(fname, par, ctype=gu.PIXEL_MASK, vers=None, verb=False) :
    """Returns specified array of calibration constants.
    
    Parameters

    - fname : full path to the hdf5 file
    - par   : psana.Event | psana.Env | float - tsec event time
    - ctype : gu.CTYPE - enumerated calibration type, e.g.: gu.PIXEL_MASK
    - vers  : int - calibration version

    Returns

    - np.array - specified array of calibration constants

    See :py:class:`DCMethods`
    """
    if not is_good_fname(fname, verb) : return None

    cs = DCStore(fname)
    cs.load()
    if verb :
        print 50*'_','\nDCStore.print_obj()' 
        cs.print_obj()

    str_ctype = gu.dic_calib_type_to_name[ctype]
    ct = cs.ctypeobj(str_ctype)
    if ct is None : return None 
    #ct.print_obj()

    tsec = dcu.par_to_tsec(par)
    #print 'XXX: get DCRange object for time = %.3f' % tsec
    cr = ct.range_for_tsec(tsec)
    #cr = ct.range_for_evt(evt)


    if cr is None : return None
    #cr.print_obj()

    cv = cr.version(vnum=vers)
    if cv is None : return None
    #cv.print_obj()

    return cv.data()

#------------------------------

def get_constants(par, env, src='Epix100a.', ctype=gu.PIXEL_MASK, calibdir=None, vers=None, verb=False) :
    """
    Returns specified array of calibration constants.

    Parameters

    - par: psana.Event | float - tsec event time
    - env: psana.Env - to get full detector name for psana.Source 
    - src: str - source short/full name, alias or full
    - ctype: gu.CTYPE - enumerated calibration type, e.g.: gu.PIXEL_MASK
    - calibdir: str - fallback path to calib dir (if xtc file is copied - calib and experiment name are lost)
    - vers: int - calibration version

    Return 

    numpy.array - array of calibratiopn constatnts.

    See :py:class:`DCMethods`
    """
    metname = sys._getframe().f_code.co_name

    str_ctype = gu.dic_calib_type_to_name[ctype]
    if verb : print '  %s.get_constants  src: %s\n  ctype: %s\n  vers: %s\n  calibdir:%s'%\
                    (metname, src, str_ctype, vers, calibdir)

    ofn = DCFileName(env, src, calibdir)
    if verb : ofn.print_attrs()

    fname = ofn.calib_file_path()

    return get_constants_from_file(fname, par, ctype, vers, verb)

#------------------------------

def delete_version_from_file(fname, par, ctype=gu.PIXEL_MASK, vers=None, cmt=None, verb=False) :
    """Delete specified version from calibration constants.
    
    Parameters
    
    - fname : full path to the hdf5 file
    - par   : psana.Event | psana.Env | float - tsec event time
    - ctype : gu.CTYPE - enumerated calibration type, e.g.: gu.PIXEL_MASK
    - vers  : int - calibration version
    - cmt   : str - comment
    - verb  : bool - verbousity

    See :py:class:`DCMethods`
    """

    metname = sys._getframe().f_code.co_name

    str_ctype = gu.dic_calib_type_to_name[ctype]
    if verb : print '  %s.delete_version_from_file:  ctype: %s  vers: %s'%\
                    (metname, str_ctype, vers)

    if not is_good_fname(fname, verb) : return None

    cs = DCStore(fname)
    cs.load()

    ct = cs.ctypeobj(str_ctype)
    if ct is None : return None 
    #ct.print_obj()

    tsec = dcu.par_to_tsec(par)
    cr = ct.range_for_tsec(tsec)
    if cr is None : return None

    v = vers if vers is not None else cr.vnum_last()

    vdel = cr.mark_version(vnum=vers, cmt=cmt)

    if verb : log.setPrintBits(02) # 0377

    cs.save()

    if verb :
        print 50*'_','\nDCStore.print_obj() after delete version %s' % str(vdel)
        cs.print_obj()
    
    return vdel

#------------------------------

def delete_version(evt, env, src='Epix100a.', ctype=gu.PIXEL_MASK, calibdir=None, vers=None, cmt=None, verb=False) :
    """Delete specified version from calibration constants.
    
    Parameters
    
    - evt : psana.Event -> event time
    - env : psana.Env -> full detector name for psana.Source 
    - src : str - source short/full name, alias or full
    - ctype : gu.CTYPE - enumerated calibration type, e.g.: gu.PIXEL_MASK
    - calibdir : str - fallback path to calib dir (if xtc file is copied - calib and experiment name are lost)
    - vers : int - calibration version
    - cmt  : str - comment
    - verb : bool - verbousity

    See :py:class:`DCMethods`
    """
    metname = sys._getframe().f_code.co_name

    str_ctype = gu.dic_calib_type_to_name[ctype]
    if verb : print '  %s.delete_version:\n  src: %s\n  ctype: %s\n  vers: %s\n  calibdir:%s'%\
                    (metname, src, str_ctype, vers, calibdir)

    ofn = DCFileName(env, src, calibdir)
    if verb : ofn.print_attrs()

    fname = ofn.calib_file_path()

    return delete_version_from_file(fname, evt, ctype, vers, cmt, verb)

#------------------------------

def delete_range_from_file(fname, ctype=gu.PIXEL_MASK, range=None, cmt=None, verb=False) :
    """Delete specified time range from calibration constants.
    
    Parameters
    
    - fname : full path to the hdf5 file
    - ctype : gu.CTYPE - enumerated calibration type, e.g.: gu.PIXEL_MASK
    - range : str - range, e.g. '1474587520-end'
    - cmt   : str - comment
    - verb  : bool - verbousity

    See :py:class:`DCMethods`
    """
    metname = sys._getframe().f_code.co_name

    str_ctype = gu.dic_calib_type_to_name[ctype]
    if verb : print '  %s.delete_range_from_file  ctype: %s  range: %s'%\
                    (metname, str_ctype, range)

    if not is_good_fname(fname, verb) : return None

    cs = DCStore(fname)
    cs.load()
    #cs.print_obj()

    ct = cs.ctypeobj(str_ctype)
    if ct is None : return None 

    rdel = ct.mark_range_for_key(range, cmt=cmt)
    if rdel is None : return None

    if verb : log.setPrintBits(02) # 0377

    cs.save()

    if verb :
        print 50*'_','\nDCStore.print_obj() after delete range %s' % rdel
        cs.print_obj()
    
    return rdel

#------------------------------

def delete_range(evt, env, src='Epix100a.', ctype=gu.PIXEL_MASK, calibdir=None, range=None, cmt=None, verb=False) :
    """Delete specified time range from calibration constants.
    
    Parameters
    
    - evt : psana.Event -> event time
    - env : psana.Env -> full detector name for psana.Source 
    - src : str - source short/full name, alias or full
    - ctype : gu.CTYPE - enumerated calibration type, e.g.: gu.PIXEL_MASK
    - calibdir : str - fallback path to calib dir (if xtc file is copied - calib and experiment name are lost)
    - range : str - range, e.g. '1474587520-end'
    - cmt   : str - comment
    - verb  : bool - verbousity

    See :py:class:`DCMethods`
    """
    metname = sys._getframe().f_code.co_name

    str_ctype = gu.dic_calib_type_to_name[ctype]
    if verb : print '  %s.delete_range  src: %s\n  ctype: %s\n  range: %s\n  calibdir:%s'%\
                    (metname, src, str_ctype, range, calibdir)

    ofn = DCFileName(env, src, calibdir)
    if verb : ofn.print_attrs()

    fname = ofn.calib_file_path()

    return delete_range_from_file(fname, ctype, range, cmt, verb)

#------------------------------

def delete_ctype_from_file(fname, ctype=gu.PIXEL_MASK, cmt=None, verb=False) :
    """Delete specified ctype from calibration constants.
    
    Parameters
    
    - fname : full path to the hdf5 file
    - ctype : gu.CTYPE - enumerated calibration type, e.g.: gu.PIXEL_MASK
    - cmt   : str - comment
    - verb  : bool - verbousity

    See :py:class:`DCMethods`
    """
    metname = sys._getframe().f_code.co_name

    str_ctype = gu.dic_calib_type_to_name[ctype]
    if verb : print '  %s.delete_ctype  ctype: %s  range: %s'%\
                    (metname, str_ctype, range)

    if not is_good_fname(fname, verb) : return None

    cs = DCStore(fname)
    cs.load()
    #cs.print_obj()

    tdel = cs.mark_ctype(str_ctype, cmt=cmt)
    if tdel is None : return None 

    if verb : log.setPrintBits(02) # 0377

    cs.save()

    if verb :
        print 50*'_','\nDCStore.print_obj() after delete ctype %s' % tdel
        cs.print_obj()
    
    return tdel

#------------------------------

def delete_ctype(evt, env, src='Epix100a.', ctype=gu.PIXEL_MASK, calibdir=None, cmt=None, verb=False) :
    """Delete specified ctype from calibration constants.
    
    Parameters
    
    - evt      : psana.Event -> event time
    - env      : psana.Env -> full detector name for psana.Source 
    - src      : str - source short/full name, alias or full
    - ctype    : gu.CTYPE - enumerated calibration type, e.g.: gu.PIXEL_MASK
    - calibdir : str - fallback path to calib dir (if xtc file is copied - calib and experiment name are lost)
    - cmt      : str - comment
    - verb     : bool - verbousity

    See :py:class:`DCMethods`
    """
    metname = sys._getframe().f_code.co_name

    str_ctype = gu.dic_calib_type_to_name[ctype]
    if verb : print '  %s.delete_ctype  src: %s\n  ctype: %s\n  range: %s\n  calibdir:%s'%\
                    (metname, src, str_ctype, range, calibdir)

    ofn = DCFileName(env, src, calibdir)
    if verb : ofn.print_attrs()

    fname = ofn.calib_file_path()

    return delete_ctype_from_file(fname, ctype, cmt, verb)

#------------------------------

def print_content_from_file(fname) :
    """Prints content of the file.
    
    Parameters
    
    - fname : str - full path to the file

    See :py:class:`DCMethods`
    """
    metname = sys._getframe().f_code.co_name

    if not is_good_fname(fname, True) : return

    cs = DCStore(fname)

    t0_sec = time()
    cs.load()
    print 'File content loading time (sec) = %.6f' % (time()-t0_sec)
    
    print 50*'_','\nDCStore.print_obj()' 
    cs.print_obj()

#------------------------------

def print_content(env, src='Epix100a.', calibdir=None) :
    """Defines the file name and prints file content.
    
    Parameters
    
    - env : psana.Env -> full detector name for psana.Source 
    - src : str - source short/full name, alias or full
    - calibdir : str - fallback path to calib dir (if xtc file is copied - calib and experiment name are lost)

    See :py:class:`DCMethods`
    """
    #metname = sys._getframe().f_code.co_name

    ofn = DCFileName(env, src, calibdir)
    fname = ofn.calib_file_path()
    ofn.print_attrs()

    print_content_from_file(fname)

#------------------------------
#------------------------------
#------------------------------
#------------------------------

def get_constants_v0(*par, **opt) :
    ofn = DCFileName(par[0], opt['src'])

#------------------------------

def test_add_constants() :
    metname = sys._getframe().f_code.co_name
    print 20*'_', '\n%s' % metname

    import numpy as np

    vers = None
    #nda  = np.zeros((32,185,388), dtype=np.float32)
    nda  = np.zeros((1000,1000), dtype=np.float32)
    pred = None
    succ = None
    cmt  = 'my comment: %s' % metname
    
    add_constants(nda, gevt, genv, gsrc, gctype, gcalibdir, vers, pred, succ, cmt, gverb)
    print '%s: constants added nda.shape=%s' % (metname, nda.shape)

#------------------------------

def test_add_constants_to_file() :
    metname = sys._getframe().f_code.co_name
    print 20*'_', '\n%s' % metname

    import numpy as np

    vers = None
    #data  = np.zeros((32,185,388), dtype=np.float32)
    data  = 11*np.ones((500,500), dtype=np.float32)
    pred = None
    succ = None
    cmt  = 'my comment: %s' % metname
    par = gevt # 1474587525.3 # gevt

    add_constants_to_file(data, gfname, par, genv, gctype, vers, pred, succ, cmt, gverb)

    print '%s: constants added data.shape=%s' % (metname, data.shape)

#------------------------------

def test_add_text() :
    metname = sys._getframe().f_code.co_name
    print 20*'_', '\n%s' % metname

    import numpy as np

    data = 'Now we are going to save\n this little piece of text'
    vers = None
    pred = 'Predecessor_name'
    succ = 'Successor_name'
    ctype = gu.GEOMETRY
    cmt  = 'my comment: %s' % metname
    
    add_constants(data, gevt, genv, gsrc, ctype, gcalibdir, vers, pred, succ, cmt, gverb)
    print '%s: text is added to the file' % (metname)

#------------------------------

def test_add_text_to_file() :
    metname = sys._getframe().f_code.co_name
    print 20*'_', '\n%s' % metname

    import numpy as np

    data = 'Now we are going to save\n this little piece of text'
    vers = None
    pred = None
    succ = None
    ctype = gu.GEOMETRY
    cmt  = 'my comment: %s' % metname
    par = gevt
    
    add_constants_to_file(data, gfname, par, genv, gctype, vers, pred, succ, cmt, gverb)
    
    print '%s: text is added to the file\n  %s' % (metname, gfname)

#------------------------------

def test_add_constants_two() :
    metname = sys._getframe().f_code.co_name
    print 20*'_', '\n%s' % metname

    import numpy as np

    vers   = None
    pred   = None
    succ   = None
    ctype1 = gu.PIXEL_STATUS
    ctype2 = gu.PIXEL_MASK
    #nda   = np.zeros((32,185,388), dtype=np.float32)
    nda1   = np.ones((1000,1000), dtype=np.float32)
    nda2   = 2 * nda1
    cmt1   = 'my comment 1: %s' % metname
    cmt2   = 'my comment 2: %s' % metname
    
    add_constants(nda1, gevt, genv, gsrc, ctype1, gcalibdir, vers, pred, succ, cmt1, gverb)
    add_constants(nda2, gevt, genv, gsrc, ctype2, gcalibdir, vers, pred, succ, cmt2, gverb)
    print '%s: constants added ctype1=%s nda1.shape=%s' % (metname, str(ctype1), nda1.shape)
    print '%s: constants added ctype2=%s nda2.shape=%s' % (metname, str(ctype2), nda2.shape)

#------------------------------

def test_get_constants() :
    metname = sys._getframe().f_code.co_name
    print 20*'_', '\n%s' % metname

    vers = None
    data = get_constants(gevt, genv, gsrc, gctype, gcalibdir, vers, gverb)

    print '%s: retrieved constants for vers %s' % (metname, str(vers))
    print 'data:\n', data
    if isinstance(data, np.ndarray) : print 'data.shape=%s' % (str(data.shape))

#------------------------------

def test_get_constants_from_file() :
    metname = sys._getframe().f_code.co_name
    print 20*'_', '\n%s' % metname

    vers = None
    data = get_constants_from_file(gfname, gevt, gctype, vers, gverb)

    print '%s: retrieved constants for vers %s' % (metname, str(vers))
    print 'data:\n', data
    if isinstance(data, np.ndarray) : print 'data.shape=%s' % (str(data.shape))

#------------------------------

def test_delete_version() :
    metname = sys._getframe().f_code.co_name
    print 20*'_', '\n%s' % metname

    vers = None # for default - last version
    cmt  = 'my comment: %s' % metname
    vdel = delete_version(gevt, genv, gsrc, gctype, gcalibdir, vers, cmt, gverb)
    print '%s: deleted version %s' % (metname, str(vdel))

#------------------------------

def test_delete_version_from_file() :
    metname = sys._getframe().f_code.co_name
    print 20*'_', '\n%s' % metname

    vers = None # for default - last version
    cmt  = 'my comment: %s' % metname
    vdel = delete_version_from_file(gfname, gevt, gctype, vers, cmt, gverb)

    print '%s: deleted version %s' % (metname, str(vdel))

#------------------------------

def test_delete_range() :
    metname = sys._getframe().f_code.co_name
    print 20*'_', '\n%s' % metname

    range = '1474587520-end'
    cmt  = 'my comment: %s' % metname
    rdel = delete_range(gevt, genv, gsrc, gctype, gcalibdir, range, cmt, gverb)
    print '%s: deleted range %s' % (metname, str(rdel))

#------------------------------

def test_delete_range_from_file() :
    metname = sys._getframe().f_code.co_name
    print 20*'_', '\n%s' % metname

    range = '1474587520-end'
    cmt  = 'my comment: %s' % metname
    rdel = delete_range_from_file(gfname, gctype, range, cmt, gverb)    
    print '%s: deleted range %s' % (metname, str(rdel))

#------------------------------

def test_delete_ctype() :
    metname = sys._getframe().f_code.co_name
    print 20*'_', '\n%s' % metname

    cmt  = 'my comment: %s' % metname
    tdel = delete_ctype(gevt, genv, gsrc, gctype, gcalibdir, cmt, gverb)
    print '%s: deleted ctype %s' % (metname, str(tdel))

#------------------------------

def test_delete_ctype_from_file() :
    metname = sys._getframe().f_code.co_name
    print 20*'_', '\n%s' % metname

    cmt  = 'my comment: %s' % metname
    tdel = delete_ctype_from_file(gfname, gctype, cmt, gverb)
    
    print '%s: deleted ctype %s' % (metname, str(tdel))

#------------------------------

def test_print_content() :
    metname = sys._getframe().f_code.co_name
    print 20*'_', '\n%s' % metname
    print_content(genv, gsrc, gcalibdir)
    print '%s is completed' % (metname)

#------------------------------

def test_print_content_from_file() :
    metname = sys._getframe().f_code.co_name
    print 20*'_', '\n%s' % metname
    print_content_from_file(gfname)
    print '%s is completed' % (metname)

#------------------------------

def test_misc() :
    print 20*'_', '\n%s' % sys._getframe().f_code.co_name

    import PSCalib.DCUtils as dcu

    print 20*'_', '\n%s:' % sys._getframe().f_code.co_name
    print 'get_enviroment(USER) : %s' % dcu.get_enviroment()
    print 'get_login()          : %s' % dcu.get_login()
    print 'get_hostname()       : %s' % dcu.get_hostname()
    print 'get_cwd()            : %s' % dcu.get_cwd()

#------------------------------

def set_parameters() :

    import psana
    import numpy as np; global np

    global genv, gevt, gsrc, gctype, gcalibdir, gverb, gfname

    #dsname  = 'exp=cxif5315:run=129'
    #dsname   = '/reg/g/psdm/detector/data_test/xtc/cxif5315-e545-r0169-s00-c00.xtc'
    #gsrc      = 'Cspad.'

    #dsname = 'exp=mfxn8316:run=11'
    dsname = '/reg/g/psdm/detector/data_test/types/0021-MfxEndstation.0-Epix100a.0.xtc'
    gsrc      = ':Epix100a.'

    gcalibdir = './calib'
    gctype    = gu.PIXEL_STATUS # gu.PIXEL_MASK, gu.PEDESTALS, etc.
    gverb     = True
    #gverb     = False

    gfname = './calib/epix100a/epix100a-3925999616-0996663297-3791650826-1232098304-0953206283-2655595777-0520093719.h5'

    ds = psana.DataSource(dsname)
    genv=ds.env()
    gevt=ds.events().next()

#------------------------------

def do_test() :
    from time import time

    set_parameters()

    #log.setPrintBits(0377)

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print 50*'_', '\nTest %s:' % tname
    t0_sec = time()
    if   tname == '0' : test_misc(); 

    elif tname ==  '1': test_add_constants()
    elif tname ==  '2': test_add_constants_two()
    elif tname ==  '3': test_add_text()
    elif tname ==  '4': test_print_content()
    elif tname ==  '5': test_get_constants()
    elif tname ==  '6': test_delete_version()
    elif tname ==  '7': test_delete_range()
    elif tname ==  '8': test_delete_ctype()

    elif tname == '11': test_add_constants_to_file()
    elif tname == '12': test_add_constants_to_file()
    elif tname == '13': test_add_text_to_file()
    elif tname == '14': test_print_content_from_file()
    elif tname == '15': test_get_constants_from_file()
    elif tname == '16': test_delete_version_from_file()
    elif tname == '17': test_delete_range_from_file()
    elif tname == '18': test_delete_ctype_from_file()

    else : print 'Not-recognized test name: %s' % tname
    msg = 'End of test %s, consumed time (sec) = %.6f' % (tname, time()-t0_sec)
    sys.exit(msg)
 
#------------------------------

if __name__ == "__main__" :
    do_test()

#------------------------------
