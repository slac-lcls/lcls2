#!/usr/bin/env python
#------------------------------
"""
Class :py:class:`CalibParsBasePnccdV1` - holds basic calibration metadata parameters for associated detector
============================================================================================================

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
  -  :py:class:`CalibParsBasePnccdV1`
  -  :py:class:`CalibParsBasePrincetonV1`
  -  :py:class:`CalibParsBaseAcqirisV1`
  -  :py:class:`CalibParsBaseImpV1`

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Author: Mikhail Dubrovin
"""
#------------------------------

class CalibParsBasePnccdV1 :

    ndim = 3 
    segs = 4
    rows = 512
    cols = 512
    size = segs*rows*cols
    shape = (segs, rows, cols)
    size_cm = 7 
    shape_cm = (size_cm,)
    cmod = (1,50,50,100,1,size,1)
        
    def __init__(self) : pass

#------------------------------

