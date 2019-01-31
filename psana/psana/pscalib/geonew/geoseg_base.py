
"""
Base class :py:class:`geoseg_base` for segment geometry description
===================================================================
:py:class:`geoseg_base` supports tentative format of geometry constants for LCLS2.

Usage::

    from psana.pscalib.geonew.utils import object_from_python_code
    s = object_from_python_code('lcls2/psana/psana/pscalib/geonew/geoseg_cspad2x1.py', 'SENS2X1')

    from psana.pscalib.geonew.geoseg_base import AXIS_X, AXIS_Y, AXIS_Z
    x_coords = s.pixel_coordinate_for_axis(AXIS_X)

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-01-30 by Mikhail Dubrovin
"""

#----------

import numpy as np
from math import sqrt

import logging
logger = logging.getLogger(__name__)

#----------

MICROMETER = 1
MILIMETER  = 1e3
METER      = 1e6

AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2
AXES = (AXIS_X, AXIS_Y, AXIS_Z)

#----------

def unit_vector(v, dtype=np.float) :
    """Returns (np.array) normalized unit vector for any array of values.
    """
    a = np.array(v, dtype=dtype)        
    alen = sqrt(sum(a*a))   
    return a/alen if alen>0 else None

def pixel_coords_from_array_of_sizes(a) :
    """Returns (np.array) 1d pixel center coordinate array form 1d pixel sizes array.
    """
    return None if a is None else np.cumsum(a) - a/2 - a[0]/2

#----------

def memorize(dic={}) :
    """Caching decorator
    """
    def deco_memorize(f) :
        def wrapper(*args, **kwargs):
            fid = id(f)
            v = dic.get(fid, None)
            if v is None :
                v = dic[fid] = f(*args, **kwargs)
            return v
        return wrapper
    return deco_memorize

#----------

class geoseg_base :
    logger.debug('in geoseg_base')

    _source   = None # (str) intended to preserve source code of the derived class
    _objname  = None # (str) intended to preserve object name from text file with code

    units    = MICROMETER
    shape    = None
    pixgrps  = None

    # default matrix-like binding to local coordinate frame
    vpix0    = (0, 0, 0)
    v_s      = (1, 0, 0)
    v_f      = (0, 1, 0)

    def __init__(self) :
        logger.debug('in geoseg_base.__init__')

    def set_source(self, s) :
        self._source = s

    def set_object_name(self, s) :
        self._objname = s

    def dz_vs_xy(self,ix,iy) :
        return 0

    def nslow(self) :
        return None if self.shape is None else self.shape[0]

    def nfast(self) :
        return None if self.shape is None else self.shape[1]

    @memorize()
    def pixel_sizes_slow(self) :
        if self.shape is None : return None
        psize_slow = np.zeros(self.nslow(), dtype=float)
        for (r0, c0, r1, c1, dr, dc, dd) in self.pixgrps : psize_slow[r0:r1+1] = dr
        return  psize_slow

    @memorize()
    def pixel_sizes_fast(self) :
        if self.shape is None : return None
        psize_fast = np.zeros(self.nfast(), dtype=float)
        for (r0, c0, r1, c1, dr, dc, dd) in self.pixgrps : psize_fast[c0:c1+1] = dc
        return psize_fast

    def pixel_size_slow(self, row=0) :
        psizes = self.pixel_sizes_slow()        
        return None if psizes is None else psizes[row]

    def pixel_size_fast(self, column=0) :
        psizes = self.pixel_sizes_fast()        
        return None if psizes is None else psizes[column]

    @memorize()
    def pixel_coordinate_slow(self) : 
        return pixel_coords_from_array_of_sizes(self.pixel_sizes_slow())

    @memorize()
    def pixel_coordinate_fast(self) :
        return pixel_coords_from_array_of_sizes(self.pixel_sizes_fast())

    @memorize()
    def pixel_coordinate_arrays(self) :
        """Returns list of [X, Y, Z] pixel coordinate arrays. 
           Consumed time is ~5ms on pcds113
        """
        cfs = (self.pixel_coordinate_fast(),\
               self.pixel_coordinate_slow())

        if not all(isinstance(v,np.ndarray) for v in cfs) : 
            logger.debug('pixel_coordinate_fast/slow - are not defined as numpy array')
            return None

        uvf, uvs = map(unit_vector, (self.v_f, self.v_s))
        gf, gs = np.meshgrid(*cfs)
        return [gf*uvf[axis] + gs*uvs[axis] + self.vpix0[axis] for axis in AXES]

    def pixel_coordinate_for_axis(self, axis=AXIS_X) :
        """Returns pixel coordinate array for specified axis.
        """
        return self.pixel_coordinate_arrays()[axis]

    def str_geoseg_code_pixgrps(self, pgfmt='(%4d, %4d, %4d, %4d, %8.2f, %8.2f, %8.2f)') :
        return '  pixgrps = (\n    %s\n  )'%\
               (',\n    '.join([pgfmt%pg for pg in self.pixgrps]))

    def str_geoseg_code_body(self) :
        """Returns (str) formatted code of the class body.
        """
        return '\n%s' % self.str_geoseg_code_pixgrps()+\
               '\n  shape = %s'   % (str(self.shape))+\
               '\n  vpix0 = %s'   % (str(self.vpix0))+\
               '\n  v_f = %s'     % (str(self.v_f))+\
               '\n  v_s = %s'     % (str(self.v_s))

    def str_geoseg_code(self, cmtext='') :
        """Returns (str) code of the object to be save in geometry file.
        """
        hat  = '\n#----------\n\n'+\
               'from geoseg_base import geoseg_base\n'+\
               'class %s(geoseg_base) :' % self.__class__.__name__
        doc  = self.__doc__ + cmtext
        body = self.str_geoseg_code_body()
        bot  = '\n#---------- SINGLETON\n'+\
               '\nglobal SENS2X1'+\
               '\nSENS2X1 = geoseg_cspad2x1()'+\
               '\n\n#----------'
        return '%s\n  """%s\n  """%s\n%s\n' % (hat, doc, body, bot)

#----------

if __name__ == "__main__" :
    print('Test method is not implemented for this module... See test_geoseg.py')

#----------
