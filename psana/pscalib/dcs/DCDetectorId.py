####!/usr/bin/env python
#------------------------------
"""
Class :py:class:`DCDetectorId` for the Detector Calibration (DC) project
========================================================================

Usage::

    # Import
    from PSCalib.DCDetectorId import id_epix, id_cspad

    # Parameters
    dsn = 'exp=cxif5315:run=169'
    # or
    dsn = '/reg/g/psdm/detector/data_test/types/0003-CxiDs2.0-Cspad.0-fiber-data.xtc'
    ds = psana.DataSource(dsn)
    env = ds.env()
    src = psana.Source('CxiDs2.0:Cspad.0') # or unique portion of the name ':Cspad.' or alias 'DsaCsPad'

    # Access methods
    ide = id_epix(env, src)
    idc = id_cspad(env, src)

See:
    * :py:class:`DCStore`
    * :py:class:`DCType`
    * :py:class:`DCRange`
    * :py:class:`DCVersion`
    * :py:class:`DCBase`
    * :py:class:`DCInterface`
    * :py:class:`DCUtils`
    * :py:class:`DCDetectorId`
    * :py:class:`DCConfigParameters`
    * :py:class:`DCFileName`
    * :py:class:`DCLogger`
    * :py:class:`DCMethods`
    * :py:class:`DCEmail`

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Author: Mikhail Dubrovin
"""

#------------------------------

from PSCalib.DCUtils import detector_full_name, psana_source

from Detector.PyDataAccess import\
     get_cspad_config_object,\
     get_cspad2x2_config_object,\
     get_epix_config_object

#------------------------------

def id_epix(env, src) :
    """Returns Epix100 Id as a string, e.g., 3925999616-0996663297-3791650826-1232098304-0953206283-2655595777-0520093719"""
    psa_src = psana_source(env, src)
    o = get_epix_config_object(env, psa_src)
    fmt2 = '%010d-%010d'
    zeros = fmt2 % (0,0)
    version = '%010d' % (o.version()) if getattr(o, "version", None) is not None else '%010d' % 0
    carrier = fmt2 % (o.carrierId0(), o.carrierId1())\
              if getattr(o, "carrierId0", None) is not None else zeros
    digital = fmt2 % (o.digitalCardId0(), o.digitalCardId1())\
              if getattr(o, "digitalCardId0", None) is not None else zeros
    analog  = fmt2 % (o.analogCardId0(), o.analogCardId1())\
              if getattr(o, "analogCardId0", None) is not None else zeros
    return '%s-%s-%s-%s' % (version, carrier, digital, analog)

#------------------------------

def id_cspad(env, src) :
    """Returns detector full name for any src, e.g., XppGon.0:Cspad2x2.0"""
    return detector_full_name(env, src)

#------------------------------

def id_det_noid(env, src) :
    """Returns detector full name for any src, e.g., XppGon.0:Cspad2x2.0"""
    return detector_full_name(env, src)

#------------------------------
#------------------------------
#------------------------------
#------------------------------
#------------------------------
#------------------------------

def test_id_epix() :
    dsn = '/reg/g/psdm/detector/data_test/types/0019-XppGon.0-Epix100a.0.xtc'
    src = 'XppGon.0:Epix100a.0'  
    ds = psana.DataSource(dsn)
    env = ds.env()
    print 20*'_', '\n%s:' % sys._getframe().f_code.co_name
    print 'dataset     : %s' % dsn
    print 'source      : %s' % src
    print 'Detector Id : %s' % id_epix(env, src)

#------------------------------

def test_id_cspad() :
    dsn = '/reg/g/psdm/detector/data_test/types/0003-CxiDs2.0-Cspad.0-fiber-data.xtc'
    src = ':Cspad.0' # 'CxiDs2.0:Cspad.0'
    ds = psana.DataSource(dsn)
    env = ds.env()
    print 20*'_', '\n%s:' % sys._getframe().f_code.co_name
    print 'dataset     : %s' % dsn
    print 'source      : %s' % src
    print 'Detector Id : %s' % id_cspad(env, src)

#------------------------------

def do_test() :
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print 50*'_', '\nTest %s:' % tname
    if   tname == '0' : test_id_epix(); test_id_cspad()
    elif tname == '1' : test_id_epix()        
    elif tname == '2' : test_id_cspad()        
    else : print 'Not-recognized test: %s' % tname
    sys.exit( 'End of test %s' % tname)

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    import psana; global psana
    do_test()

#------------------------------
