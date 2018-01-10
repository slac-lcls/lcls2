#!/usr/bin/env python
#------------------------------
"""
:py:class:`CalibParsStore` - a factory class/method
===================================================

Switches between different device-dependent calibration constants using :py:class:`CalibPars` interface.

Usage::

    # Import
    from PSCalib.CalibParsStore import cps
    from PSCalib.GlobalUtils import *

    # Initialization
    calibdir = env.calibDir()  # or e.g. '/reg/d/psdm/<INS>/<experiment>/calib'
    group = None               # or e.g. 'CsPad::CalibV1'
    source = 'Camp.0:pnCCD.1'
    runnum = 10                # or e.g. evt.run()
    pbits = 255
    o = cps.Create(calibdir, group, source, runnum, pbits)

    # or using different list of parameters to access calibration from hdf5 DCS file:
    o = cps.CreateForEvtEnv(self, calibdir, group, source, evt, env, pbits=0)

    # Access methods
    nda = o.pedestals()
    nda = o.pixel_status()
    nda = o.pixel_datast()
    nda = o.pixel_rms()
    nda = o.pixel_mask()
    nda = o.pixel_gain()
    nda = o.pixel_offset()
    nda = o.pixel_bkgd()
    nda = o.common_mode()

    status = o.status(ctype=PEDESTALS) # see list of ctypes in :py:class:`GlobalUtils`
    shape  = o.shape(ctype)
    size   = o.size(ctype)
    ndim   = o.ndim(ctype)

Methods:
  -  :py:meth:`Create`
  -  :py:meth:`CreateForEvtEnv`

See:
  -  :py:class:`GenericCalibPars`
  -  :py:class:`GlobalUtils`
  -  :py:class:`CalibPars`
  -  :py:class:`CalibParsStore` 
  -  :py:class:`CalibParsBaseAndorV1`
  -  :py:class:`CalibParsBaseAndor3dV1`
  -  :py:class:`CalibParsBaseCameraV1`
  -  :py:class:`CalibParsBaseCSPad2x2V1`
  -  :py:class:`CalibParsBaseCSPadV1`
  -  :py:class:`CalibParsBaseEpix100aV1`
  -  :py:class:`CalibParsBaseEpix10kaV1`
  -  :py:class:`CalibParsBasePnccdV1`
  -  :py:class:`CalibParsBasePrincetonV1`
  -  :py:class:`CalibParsBaseAcqirisV1`
  -  :py:class:`CalibParsBaseImpV1`

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

Created: 2013-03-08 by Mikhail Dubrovin
"""

import sys

#------------------------------

import PSCalib.GlobalUtils            as gu
from PSCalib.GenericCalibPars         import GenericCalibPars

from PSCalib.CalibParsBaseAndorV1     import CalibParsBaseAndorV1    
from PSCalib.CalibParsBaseAndor3dV1   import CalibParsBaseAndor3dV1    
from PSCalib.CalibParsBaseCameraV1    import CalibParsBaseCameraV1   
from PSCalib.CalibParsBaseCSPad2x2V1  import CalibParsBaseCSPad2x2V1 
from PSCalib.CalibParsBaseCSPadV1     import CalibParsBaseCSPadV1    
from PSCalib.CalibParsBaseEpix100aV1  import CalibParsBaseEpix100aV1 
from PSCalib.CalibParsBaseEpix10kaV1  import CalibParsBaseEpix10kaV1 
from PSCalib.CalibParsBasePnccdV1     import CalibParsBasePnccdV1    
from PSCalib.CalibParsBasePrincetonV1 import CalibParsBasePrincetonV1
from PSCalib.CalibParsBaseAcqirisV1   import CalibParsBaseAcqirisV1
from PSCalib.CalibParsBaseImpV1       import CalibParsBaseImpV1
from PSCalib.CalibParsBaseJungfrauV1  import CalibParsBaseJungfrauV1

#------------------------------

class CalibParsStore() :
    """Factory class for CalibPars object of different detectors"""

#------------------------------

    def __init__(self) :
        self.name = self.__class__.__name__
        
#------------------------------

    def Create(self, calibdir, group, source, runnum, pbits=0, fnexpc=None, fnrepo=None, tsec=None) :
        """ Factory method

            Parameters

            - calibdir : string - calibration directory, ex: /reg/d/psdm/AMO/amoa1214/calib
            - group    : string - group, ex: PNCCD::CalibV1
            - source   : string - data source, ex: Camp.0:pnCCD.0
            - runnum   : int    - run number, ex: 10
            - pbits=0  : int    - print control bits, ex: 255

            Returns

            - GenericCalibPars object
        """        

        dettype = gu.det_type_from_source(source)
        grp = group if group is not None else gu.dic_det_type_to_calib_group[dettype]

        if pbits : print '%s: Detector type = %d: %s' % (self.name, dettype, gu.dic_det_type_to_name[dettype])

        cbase = None
        if   dettype ==  gu.CSPAD     : cbase = CalibParsBaseCSPadV1()
        elif dettype ==  gu.CSPAD2X2  : cbase = CalibParsBaseCSPad2x2V1() 
        elif dettype ==  gu.PNCCD     : cbase = CalibParsBasePnccdV1()    
        elif dettype ==  gu.PRINCETON : cbase = CalibParsBasePrincetonV1()
        elif dettype ==  gu.ANDOR3D   : cbase = CalibParsBaseAndor3dV1()    
        elif dettype ==  gu.ANDOR     : cbase = CalibParsBaseAndorV1()    
        elif dettype ==  gu.EPIX100A  : cbase = CalibParsBaseEpix100aV1() 
        elif dettype ==  gu.EPIX10KA  : cbase = CalibParsBaseEpix10kaV1() 
        elif dettype ==  gu.JUNGFRAU  : cbase = CalibParsBaseJungfrauV1()    
        elif dettype ==  gu.ACQIRIS   : cbase = CalibParsBaseAcqirisV1() 
        elif dettype ==  gu.IMP       : cbase = CalibParsBaseImpV1() 
        elif dettype in (gu.OPAL1000,\
                         gu.OPAL2000,\
                         gu.OPAL4000,\
                         gu.OPAL8000,\
                         gu.TM6740,\
                         gu.ORCAFL40,\
                         gu.FCCD960,\
                         gu.QUARTZ4A150,\
                         gu.RAYONIX,\
                         gu.FCCD,\
                         gu.TIMEPIX,\
                         gu.FLI,\
                         gu.ZYLA,\
                         gu.EPICSCAM,\
                         gu.PIMAX) : cbase = CalibParsBaseCameraV1()

        else :
            print '%s: calibration is not implemented data source "%s"' % (self.__class__.__name__, source)
            #raise IOError('Calibration parameters for source: %s are not implemented in class %s' % (source, self.__class__.__name__))
        return GenericCalibPars(cbase, calibdir, grp, source, runnum, pbits, fnexpc, fnrepo, tsec)

#------------------------------

    def CreateForEvtEnv(self, calibdir, group, source, par, env, pbits=0) :
        """ Factory method
            This method makes access to the calibration store with fallback access to hdf5 file.

            Parameters

            - calibdir : string - calibration directory, ex: /reg/d/psdm/AMO/amoa1214/calib
            - group    : string - group, ex: PNCCD::CalibV1
            - source   : string - data source, ex: Camp.0:pnCCD.0
            - par      : int runnum or psana.Event - is used to get run number
            - env      : psana.Env   - environment object - is used to retrieve file name to get dataset time to retrieve DCRange
            - pbits=0  : int         - print control bits, ex: 255

            Returns

            - GenericCalibPars object
        """
        from PSCalib.DCFileName import DCFileName
        from PSCalib.DCUtils import env_time  #, evt_time

        runnum = par if isinstance(par, int) else par.run()

        #fnexpc, fnrepo, tsec = None, None, None

        ofn = DCFileName(env, source, calibdir)
        if pbits & 512 : ofn.print_attrs()
        fnexpc = ofn.calib_file_path()
        fnrepo = ofn.calib_file_path_repo()
        tsec = env_time(env)

        #if True :
        if pbits :
            print '%s.CreateForEvtEnv: for tsec: %s' % (self.name, str(tsec))
            print '  expected hdf5 file name repo : "%s"' % (fnrepo)
            print '  expected hdf5 file name local: "%s"' % (fnexpc)

        return self.Create(calibdir, group, source, runnum, pbits, fnexpc, fnrepo, tsec)

#------------------------------

cps = CalibParsStore()

#------------------------------
#------------------------------
#----------- TEST -------------
#------------------------------
#------------------------------

import numpy as np

def print_nda(nda, cmt='') :
    arr = nda if isinstance(nda, np.ndarray) else np.array(nda) 
    str_arr = str(arr) if arr.size<5 else str(arr.flatten()[0:5])
    print '%s %s: shape=%s, size=%d, dtype=%s, data=%s' % \
          (cmt, type(nda), str(arr.shape), arr.size, str(arr.dtype), str_arr)

#------------------------------

def test_Create(tname='0') :

    calibdir = '/reg/d/psdm/CXI/cxif5315/calib'
    group    = None # will be substituted from dictionary or 'CsPad::CalibV1' 
    source   = 'CxiDs2.0:Cspad.0'
    runnum   = 60
    pbits    = 0
 
    if(tname=='0') :
        o = cps.Create(calibdir, group, source, runnum, pbits)
        o.print_attrs()

        print_nda(o.pedestals(),    'pedestals')
        print_nda(o.pixel_rms(),    'pixel_rms')
        print_nda(o.pixel_mask(),   'pixel_mask')
        print_nda(o.pixel_status(), 'pixel_status')
        print_nda(o.pixel_gain(),   'pixel_gain')
        print_nda(o.common_mode(),  'common_mode')
        print_nda(o.pixel_bkgd(),   'pixel_bkgd') 
        print_nda(o.shape(),        'shape')
 
        print 'size=%d' % o.size()
        print 'ndim=%d' % o.ndim()

        statval = o.status(gu.PEDESTALS)
        print 'status(PEDESTALS)=%d: %s' % (statval, gu.dic_calib_status_value_to_name[statval])

        statval = o.status(gu.PIXEL_GAIN)
        print 'status(PIXEL_GAIN)=%d: %s' % (statval, gu.dic_calib_status_value_to_name[statval])
 
    else : print 'Non-expected arguments: sys.argv = %s use 1,2,...' % sys.argv

#------------------------------

def test_CreateForEvtEnv(tname='0') :

    from psana import DataSource
 
    dsname = 'exp=cxif5315:run=129'
    source = 'CxiDs2.0:Cspad.0'

    if tname == '2' :
        dsname = '/reg/g/psdm/detector/data_test/types/0019-XppGon.0-Epix100a.0.xtc'
        source = 'XppGon.0:Epix100a.0'

    ds = DataSource(dsname)
    env = ds.env()

    calibdir = env.calibDir() # '/reg/d/psdm/CXI/cxif5315/calib'
    group    = None # will be substituted from dictionary or 'CsPad::CalibV1' 
    runnum   = 160
    pbits    = 0

    par      = runnum

    o = cps.CreateForEvtEnv(calibdir, group, source, par, env, pbits)
    o.print_attrs()

#------------------------------

if __name__ == "__main__" :
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print 50*'_', '\nTest %s:' % tname
    if   tname == '0' : test_Create(tname)
    elif tname in ('1','2') : test_CreateForEvtEnv(tname)
    else : print 'Non-implemented test: %s' % tname
    sys.exit( 'End of %s test.' % sys.argv[0])

#------------------------------
