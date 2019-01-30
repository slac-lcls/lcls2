
"""
Example of calibration constants for CSPAD geometry description
===============================================================
Tentative format of geometry constants for LCLS2.
This file is used for test purpose only;
presumably, files like this should be located in text format in calibration DB.

Usage::

    from psana.pscalib.geonew.utils import object_from_python_code
    geometry_cspad = object_from_python_code('lcls2/psana/psana/pscalib/geonew/geometry_cspad.py', 'geometry_cspad')
    g = geometry_cspad()
    print('dir(g):\n', dir(g))
    print('str_geometry_code():\n', g.str_geometry_code(cmtext='Additional comment comes here.\n  '))

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-01-12 by Mikhail Dubrovin
"""

from geoseg_base import geoseg_base

class geoseg_cspad2x1(geoseg_base) :
    """
    CSPAD2X1 segment geometry description constants.
    Created on 2019-01-10 for LCLS2 by Mikhail Dubrovin
    """

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

global SENS2X1
SENS2X1 = geoseg_cspad2x1()

#----------
#----------

from geometry import geometry_access
class geometry_cspad(geometry_access) :
  """
  CSPAD detector geometry description constants.
  Constants were taken from one of the LCLS1 cspad geometry files. USE FOR TEST PURPOSE ONLY.
  Created on 2019-01-10 for LCLS2 by Mikhail Dubrovin
  """
  QUAD = 'QUAD'
  CSPAD = 'CSPAD'
  # PARENT IND       OBJ   IND    X0[um]   Y0[um]   Z0[um]    ROT-Z  ROT-Y  ROT-X      TILT-Z    TILT-Y    TILT-X 
  detparts = (
    (QUAD,  0,    SENS2X1,  0,     21736,   32910,       0,       0,     0,     0,    0.15393,  0.00000,  0.00000),
    (QUAD,  0,    SENS2X1,  1,     21764,   10525,       0,       0,     0,     0,    0.08944,  0.00000,  0.00000),
    (QUAD,  0,    SENS2X1,  2,     33135,   68357,       0,     270,     0,     0,    0.07829,  0.00000,  0.00000),
    (QUAD,  0,    SENS2X1,  3,     10548,   68345,       0,     270,     0,     0,    0.05061,  0.00000,  0.00000),
    (QUAD,  0,    SENS2X1,  4,     68567,   56864,       0,     180,     0,     0,   -0.07434,  0.00000,  0.00000),
    (QUAD,  0,    SENS2X1,  5,     68641,   79593,       0,     180,     0,     0,   -0.17300,  0.00000,  0.00000),
    (QUAD,  0,    SENS2X1,  6,     77801,   21584,       0,     270,     0,     0,   -0.07237,  0.00000,  0.00000),
    (QUAD,  0,    SENS2X1,  7,     54887,   21619,       0,     270,     0,     0,   -0.01447,  0.00000,  0.00000),
                                                                                                           
    (QUAD,  1,    SENS2X1,  0,     21790,   33346,       0,       0,     0,     0,   -0.32330,  0.00000,  0.00000),
    (QUAD,  1,    SENS2X1,  1,     21785,   10451,       0,       0,     0,     0,   -0.05394,  0.00000,  0.00000),
    (QUAD,  1,    SENS2X1,  2,     33453,   68428,       0,     270,     0,     0,   -0.03026,  0.00000,  0.00000),
    (QUAD,  1,    SENS2X1,  3,     10627,   68302,       0,     270,     0,     0,    0.03485,  0.00000,  0.00000),
    (QUAD,  1,    SENS2X1,  4,     68694,   56780,       0,     180,     0,     0,   -0.02960,  0.00000,  0.00000),
    (QUAD,  1,    SENS2X1,  5,     68693,   79283,       0,     180,     0,     0,   -0.04735,  0.00000,  0.00000),
    (QUAD,  1,    SENS2X1,  6,     77507,   21091,       0,     270,     0,     0,    0.36448,  0.00000,  0.00000),
    (QUAD,  1,    SENS2X1,  7,     54786,   21054,       0,     270,     0,     0,   -0.05855,  0.00000,  0.00000),
                                                                                                           
    (QUAD,  2,    SENS2X1,  0,     21762,   33294,       0,       0,     0,     0,    0.11371,  0.00000,  0.00000),
    (QUAD,  2,    SENS2X1,  1,     21805,   10433,       0,       0,     0,     0,   -0.13548,  0.00000,  0.00000),
    (QUAD,  2,    SENS2X1,  2,     33267,   68360,       0,     270,     0,     0,   -0.01447,  0.00000,  0.00000),
    (QUAD,  2,    SENS2X1,  3,     10584,   68343,       0,     270,     0,     0,   -0.06908,  0.00000,  0.00000),
    (QUAD,  2,    SENS2X1,  4,     68488,   57005,       0,     180,     0,     0,   -0.05263,  0.00000,  0.00000),
    (QUAD,  2,    SENS2X1,  5,     68432,   79403,       0,     180,     0,     0,    0.11709,  0.00000,  0.00000),
    (QUAD,  2,    SENS2X1,  6,     77293,   21711,       0,     270,     0,     0,    0.08747,  0.00000,  0.00000),
    (QUAD,  2,    SENS2X1,  7,     54717,   21678,       0,     270,     0,     0,   -0.05798,  0.00000,  0.00000),
                                                                                                           
    (QUAD,  3,    SENS2X1,  0,     21666,   32901,       0,       0,     0,     0,    0.14015,  0.00000,  0.00000),
    (QUAD,  3,    SENS2X1,  1,     21764,   10451,       0,       0,     0,     0,   -0.00132,  0.00000,  0.00000),
    (QUAD,  3,    SENS2X1,  2,     33256,   68605,       0,     270,     0,     0,    0.02302,  0.00000,  0.00000),
    (QUAD,  3,    SENS2X1,  3,     10622,   68562,       0,     270,     0,     0,   -0.10985,  0.00000,  0.00000),
    (QUAD,  3,    SENS2X1,  4,     68505,   56808,       0,     180,     0,     0,    0.03618,  0.00000,  0.00000),
    (QUAD,  3,    SENS2X1,  5,     68490,   79392,       0,     180,     0,     0,    0.14864,  0.00000,  0.00000),
    (QUAD,  3,    SENS2X1,  6,     77964,   21507,       0,     270,     0,     0,    0.04079,  0.00000,  0.00000),
    (QUAD,  3,    SENS2X1,  7,     54602,   21509,       0,     270,     0,     0,    0.12957,  0.00000,  0.00000),
                                                                                                                     
    (CSPAD, 0,    QUAD,     0,     -4500,   -4500,       0,      90,     0,     0,    0.00000,  0.00000,  0.00000),
    (CSPAD, 0,    QUAD,     1,     -4500,    4500,       0,       0,     0,     0,    0.00000,  0.00000,  0.00000),
    (CSPAD, 0,    QUAD,     2,      4500,    4500,       0,     270,     0,     0,    0.00000,  0.00000,  0.00000),
    (CSPAD, 0,    QUAD,     3,      4500,   -4500,       0,     180,     0,     0,    0.00000,  0.00000,  0.00000),

    ('IP',    0,  CSPAD,    0,         0,       0,  100000,       0,     0,     0,    0.00000,  0.00000,  0.00000),
  )

#----------
