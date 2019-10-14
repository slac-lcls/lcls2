
#--------

import numpy as np

#--------

def test_01() :
    from ndarray import test_nda_fused_v2
    print(50*'_', '\nTest of templated function test_nda_fused_v2')
    nda = np.arange(10, 2, -1, dtype=np.int16)
    nda.shape = (2,4)
    test_nda_fused_v2(nda)

#--------

def test_templ(nda) :
    from ndarray import test_nda_fused
    test_nda_fused(nda)
    print('In test_nda_fused returned array:\n', nda)

#--------

def test_02() :
    print(50*'_', '\nTest of templated function test_nda')
    test_templ(1*np.ones((2,3), dtype=np.double))
    test_templ(2*np.ones((2,4), dtype=np.int16))
    test_templ(3*np.ones((2,5), dtype=np.uint16))

#--------

def test_03() :
    print(50*'_', '\nTest of py_ctest_vector')
    from ndarray import py_ctest_vector

    py_ctest_vector(np.arange(0, 10, 1, dtype=np.double))
    py_ctest_vector(np.arange(1, 10, 1, dtype=np.float))
    py_ctest_vector(np.arange(10, 1,-1, dtype=np.int))

#--------

def test_04() :
    print(50*'_', '\nTest of py_ndarray')
    from ndarray import py_ndarray_double
    a = py_ndarray_double()
    a.set_nda(np.ones((2,3), dtype=np.double))

#--------

def usage(tname):
    s = '\nUsage: python psana/psana/hexanode/examples/ex-00-np-to-cpp-ndarray.py <test-number>'
    if tname in ('0',)    : s+='\n 0 - test ALL'
    if tname in ('0','1') : s+='\n 1 - templated function test_nda_fused_v2'
    if tname in ('0','2') : s+='\n 2 - templated function test_nda'
    if tname in ('0','3') : s+='\n 3 - test of py_ctest_vector'
    if tname in ('0','4') : s+='\n 4 - test of py_ndarray'
    return s

#--------

if __name__ == "__main__" :
    import sys; global sys
    import numpy as np; global np
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    s = 'End of Test %s' % tname
    print('%s' % usage(tname))
    print(50*'_', '\nTest %s' % tname)
    if tname in ('0','1') : test_01()
    if tname in ('0','2') : test_02()
    if tname in ('0','3') : test_03()
    if tname in ('0','4') : test_04()
    print('%s' % usage(tname))
    sys.exit(s)

#--------
