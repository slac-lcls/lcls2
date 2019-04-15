#------------------------------
"""
Class :py:class:`TDNodeRecord` helps to retreive and use peak data in processing
================================================================================

Usage::

    # Import
    from pyimgalgos.TDNodeRecord import TDNodeRecord

    # make object
    rec = TDNodeRecord(line)

    # access record attributes
    index, beta, omega, h, k, l, dr, R, qv, qh, qt, ql, P =\
    rec.index, rec.beta, rec.omega, rec.h, rec.k, rec.l, rec.dr, rec.R, rec.qv, rec.qh, rec.qt, rec.ql, rec.P
    line = rec.line

    # print attributes
    rec.print_short()

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

See:
  - :py:class:`Graphics`
  - :py:class:`GlobalGraphics`
  - :py:class:`GlobalUtils`
  - :py:class:`NDArrGenerators`
  - :py:class:`Quaternion`
  - :py:class:`FiberAngles`
  - :py:class:`FiberIndexing`
  - :py:class:`PeakData`
  - :py:class:`PeakStore`
  - :py:class:`TDCheetahPeakRecord`
  - :py:class:`TDFileContainer`
  - :py:class:`TDGroup`
  - :py:class:`TDMatchRecord`
  - :py:class:`TDNodeRecord`
  - :py:class:`TDPeakRecord`
  - :py:class:`HBins`
  - :py:class:`HPolar`
  - :py:class:`HSpectrum`
  - :py:class:`RadialBkgd`
  - `Analysis of data for cxif5315 <https://confluence.slac.stanford.edu/display/PSDMInternal/Analysis+of+data+for+cxif5315>`_.

Created in 2015 by Mikhail Dubrovin
"""

#--------------------------------

import math
#import numpy as np
#from time import strftime, localtime #, gmtime

#------------------------------

class TDNodeRecord :

    def __init__(sp, line) : # , sigma_q = 0.000484) :  
        """Parse the string of parameters to values

        # sigma_q   = 0.000484 1/A (approximately pixel size/sample-to-detector distance = 100um/100mm)

        # beta 0.00  omega 52.50 degree
        # index    beta   omega   h   k   l   dr[1/A]  R(h,k,l)   qv[1/A]   qh[1/A]   qt[1/A]   ql[1/A]   P(omega)
           106     0.00   52.50  -2  -1   0 -0.002756  0.123157  0.000000 -0.123478 -0.122470 -0.015745   0.165321
        """
        sp.fields_nr = line.rstrip('\n').split()

        if len(sp.fields_nr) == 13 :
          s_index, s_beta, s_omega, s_h, s_k, s_l, s_dr, s_R, s_qv, s_qh, s_qt, s_ql, s_P = sp.fields_nr[0:13]
          sp.qt, sp.ql = float(s_qt), float(s_ql)
        else :
          s_index, s_beta, s_omega, s_h, s_k, s_l, s_dr, s_R, s_qv, s_qh, s_P = sp.fields_nr[0:11]
          sp.qt, sp.ql = None, None

        sp.index, sp.beta, sp.omega = int(s_index), float(s_beta), float(s_omega)
        sp.h, sp.k, sp.l = int(s_h), int(s_k), int(s_l)
        sp.dr, sp.R, sp.qv, sp.qh, sp.P = float(s_dr), float(s_R), float(s_qv), float(s_qh), float(s_P) 

        sp.line  = line
        sp.empty = sp.empty_line()
        
#------------------------------
    
    def empty_line(sp) :
        fmt = '%6d  %7.2f %7.2f %3d %3d %3d %9.6f %9.6f %9.6f %9.6f %9.6f %9.6f %9.6f'
        z = 0
        return fmt % (z,z,z,z,z,z,z,z,z,z,z,z,z)       

#------------------------------

    def print_short(sp) :
        """Prints short subset of data
        """    
        print '%6d  %7.2f  %7.2f  %2d %2d %2d    %9.6f  %9.6f  %9.6f  %9.6f  %9.6f  %9.6f  %9.6f' % \
              (sp.index, sp.beta, sp.omega, sp.h, sp.k, sp.l, sp.dr, sp.R, sp.qv, sp.qh, sp.qt, sp.ql, sp.P)   

#------------------------------
#--------------------------------
#-----------  TEST  -------------
#--------------------------------

if __name__ == "__main__" :
    pass

#--------------------------------
