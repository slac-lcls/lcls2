#!/usr/bin/env python
#------------------------------
"""
Class :py:class:`CalibParsBaseCSPadV1` holds basic calibration metadata parameters for associated detector
==========================================================================================================

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

class CalibParsBaseCSPadV1 :

    ndim = 3 
    quads= 4 
    segs = 8 
    rows = 185 
    cols = 388 
    size = quads*segs*rows*cols
    shape = (quads*segs, rows, cols)
    size_cm = 4 
    shape_cm = (size_cm,)
    cmod = (1, 25, 25, 100)
        
    def __init__(self) : pass

#------------------------------

