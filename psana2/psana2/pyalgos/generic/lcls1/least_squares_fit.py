#------------------------------
"""GUI for configuration of detector object.
   Created: 2017-06-23
   Author : Mikhail Dubrovin

Usage ::
   from expmon.least_squares_fit import *
   a, b, cx0, cx1, cx2, cxy, cy1 = least_squares_fit(x,y,e)
"""
#------------------------------

import numpy as np

def least_squares_fit(x,y,e) :
    """Fits data points x,y with errors e to the line y=ax+b.
       Returns a, b, cx0, cx1, cx2, cxy, cy1
    """
    ZERO = 1e-12

    xa = np.array(x, dtype=np.float64)
    ya = np.array(y, dtype=np.float64)
    ea = np.array(e, dtype=np.float64)

    s2 = ea*ea
    s2 = np.select((s2>0,), (s2,), default=ZERO)
    os2 = 1/s2
    
    cx0 = (os2).mean()
    cx1 = (xa*os2).mean()
    cx2 = (xa*xa*os2).mean()
    cxy = (xa*ya*os2).mean()
    cy1 = (ya*os2).mean()

    d =  cx2*cx0 - cx1*cx1
    a = (cxy*cx0 - cy1*cx1)/d if d else None
    b = (cx2*cy1 - cx1*cxy)/d if d else None

    return a, b, cx0, cx1, cx2, cxy, cy1

#------------------------------
