#!/usr/bin/env python
#------------------------------
"""
:py:class:`CalibParsBaseEpix10kaV1` - holds basic calibration metadata parameters for associated detector
=========================================================================================================

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

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Author: Mikhail Dubrovin
"""
#------------------------------

class CalibParsBaseEpix10kaV1 :

    ndim = 1 
    rows = 352 
    cols = 384 
    size = rows*cols
    shape = (rows, cols)
    size_cm = 16 
    shape_cm = (size_cm,)
    cmod = (3, 100, 100, 384, 0,0,0,0, 0,0,0,0, 0,0,0,0)
    # 3-median, 100-max threshold, 100-max correction, 384-pix group size
         
    def __init__(self) : pass

#------------------------------

