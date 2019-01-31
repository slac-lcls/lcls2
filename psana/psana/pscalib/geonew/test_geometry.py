
"""
Module :py:class:`test_geometry` is designed as a test for geometry class
=========================================================================

Use command::
    python lcls2/psana/psana/pscalib/geonew/test_geometry.py

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-01-30 by Mikhail Dubrovin
"""

#----------

def test_import_geometry() :

    from psana.pscalib.geonew.geometry_cspad import geometry_cspad

    g = geometry_cspad()

    #print 'o.detparts'; for rec in o.detparts : print rec

    print('docstring:', g.__doc__)

    #print 'print_geometry_objects:\n'; o.print_geometry_objects()
    #print 'TEST SENS2X1.shape:', SENS2X1.shape

    print('str_geometry_code():\n', g.str_geometry_code(cmtext='Additional comment comes here.\n  '))

#----------

def test_load_geometry() :
    from psana.pscalib.geonew.utils import object_from_python_code
    geometry_cspad = object_from_python_code('lcls2/psana/psana/pscalib/geonew/geometry_cspad.py', 'geometry_cspad')

    g = geometry_cspad()

    print('docstring:', g.__doc__)
    print('dir(g):\n', dir(g))
    print('str_geometry_code():\n', g.str_geometry_code(cmtext='Additional comment comes here.\n  '))

    #g.print_geometry_objects()
    geo0 = g.list_of_geos[0]
    geoseg = getattr(geo0, 'oname', None)

    #import inspect # DOES NOT WORK
    #print(inspect.getsource(geo0[3]))

    print(geoseg.str_geoseg_code(cmtext='my test comment is here'))

#----------

if __name__ == "__main__" :
    #test_import_geometry()
    test_load_geometry()

#----------
