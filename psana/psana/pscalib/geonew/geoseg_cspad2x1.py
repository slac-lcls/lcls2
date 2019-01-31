"""
Example of calibration constants for CSPAD2X1 segment geometry description
==========================================================================
Tentative format of geometry constants for LCLS2.
This file is used for test purpose only;
presumably, files like this should be located in text format in calibration DB.

Usage::
    from psana.pscalib.geonew.utils import object_from_python_code
    s = object_from_python_code('lcls2/psana/psana/pscalib/geonew/geoseg_cspad2x1.py', 'SENS2X1')

    from psana.pscalib.geonew.geoseg_base import AXIS_X, AXIS_Y, AXIS_Z
    x_coords = s.pixel_coordinate_for_axis(AXIS_X)

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-01-12 by Mikhail Dubrovin
"""

#----------

from geoseg_base import geoseg_base

class geoseg_cspad2x1(geoseg_base) :

    shape = (185,388) # DAQ-like data array shape of segment

    dim_s = 109.92    
    dim_f = 109.92
    dim_d = 400.00
    dim_w = dim_f*2.5

    # Description of uniform pixel groups in segment.
    # Group of pixels is defined by the 2d-block corner indexes (s0, f0) and (s1, f1)
    # pixgrps=[(s0, f0,  s1, f1,  dim_s, dim_f, dim_d),...]
    pixgrps = ((0,   0, 184, 192, dim_s, dim_f, dim_d),
               (0, 193, 184, 194, dim_s, dim_w, dim_d),
               (0, 195, 184, 387, dim_s, dim_f, dim_d),
              )

    # Binding of 2-d pixel coordinate array(matrix) to the local coordinate frame.
    vpix0 = (-42868.8/2, 20225.28/2, 0) # (x,y,z) vector from origin to pixel with indexes (0,0).
    v_f   = (1, 0, 0)                   # (x,y,z) vector along column (fast) index ascending
    v_s   = (0,-1, 0)                   # (x,y,z) vector along row    (slow) index ascending

#---------- SINGLETON

SENS2X1 = geoseg_cspad2x1()

#----------
