
"""
Module :py:class:`test_geoseg` is designed as a test for geoseg_base class
==========================================================================

Use command::
    python lcls2/psana/psana/pscalib/geonew/test_geoseg.py

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-01-30 by Mikhail Dubrovin
"""

#----------

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(name)s %(lineno)d %(message)s', level=logging.DEBUG) # INFO) # DEBUG

#----------

def print_line() : print(100*'_')

#----------

def test_import_geoseg_cspad2x1() :

    from psana.pscalib.geonew.geoseg_cspad2x1 import seg_cspad2x1
    from psana.pscalib.geonew.geoseg_base import AXIS_X, AXIS_Y, AXIS_Z

    s = seg_cspad2x1

    print('pixel_sizes_slow()[0:5]    :', s.pixel_sizes_slow()[0:5], '...')
    print('pixel_sizes_fast()[190:195]:', s.pixel_sizes_fast()[190:195], '...')
    print('pixel_coordinate_fast():\n', s.pixel_coordinate_fast())# 42868.8/2)
    print('pixel_coordinate_slow():\n', s.pixel_coordinate_slow())# 20225.28/2)
    print('nslow:', s.nslow())
    print('nfast:', s.nfast())
    print('shape:', s.shape)
    print('uvf:', s.v_f)
    print('uvs:', s.v_s)

    ca = s.pixel_coordinate_for_axis(AXIS_X)
    print('cx.shape:', ca.shape)
    print('cx[:5, :3]:\n', ca[:5, :3])
    print('cx[:5,-3:]:\n', ca[:5,-3:])

    ca = s.pixel_coordinate_for_axis(AXIS_Y)
    print('cy.shape:', ca.shape)
    print('cy[:3, :5]:\n', ca[:3, :5])
    print('cy[-3:,:5]:\n', ca[-3:,:5])

    ca = s.pixel_coordinate_for_axis(AXIS_Z)
    print('cz.shape:', ca.shape)
    print('cz[:3, :5]:\n', ca[:3, :5])
    print('cz[-3:,:5]:\n', ca[-3:,:5])

    print('pixel_size_slow(10) ', s.pixel_size_slow(10))
    print('pixel_size_fast(20) ', s.pixel_size_fast(20))
    print('pixel_size_fast(193)', s.pixel_size_fast(193))
    print('pixel_size_fast(194)', s.pixel_size_fast(194))
    print('pixel_size_fast(195)', s.pixel_size_fast(195))

#----------

def test_load_geoseg_cspad2x1() :
    print('in test_load_geoseg_cspad2x1')
    from psana.pscalib.geonew.geoseg_base import AXIS_X, AXIS_Y, AXIS_Z

    from psana.pscalib.geonew.utils import object_from_python_code
    s = object_from_python_code('lcls2/psana/psana/pscalib/geonew/geoseg_cspad2x1.py', 'SENS2X1')

    print('docstring:', s.__doc__)
    print('dir(s):\n', dir(s))
    #print('help:'; help(s))

    print('pixel_sizes_slow()[0:5]    :', s.pixel_sizes_slow()[0:5], '...')
    print('pixel_sizes_fast()[190:195]:', s.pixel_sizes_fast()[190:195], '...')

    ca = s.pixel_coordinate_for_axis(AXIS_X)
    print('cx.shape:', ca.shape)
    print('cx[:5, :3]:\n', ca[:5, :3])
    print('cx[:5,-3:]:\n', ca[:5,-3:])

    ca = s.pixel_coordinate_for_axis(AXIS_Y)
    print('cy.shape:', ca.shape)
    print('cy[:3, :5]:\n', ca[:3, :5])
    print('cy[-3:,:5]:\n', ca[-3:,:5])

#----------

if __name__ == "__main__" :
    #print_line(); test_import_geoseg_cspad2x1())
    print_line(); test_load_geoseg_cspad2x1()
    print_line()

#----------
