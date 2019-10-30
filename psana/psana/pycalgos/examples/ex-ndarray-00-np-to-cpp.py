#!/usr/bin/env python
#--------

import numpy as np
from psana.pyalgos.generic.NDArrUtils import print_ndarr

#--------

def test_templ(nda) :
    from ndarray import test_nda_fused
    test_nda_fused(nda)
    print('In test_nda_fused returned array:\n', nda)

#--------

def test_01() :
    print(50*'_', '\nTest of templated function test_nda_fused')
    test_templ(1*np.ones((2,3), dtype=np.double))
    test_templ(2*np.ones((2,4), dtype=np.int16))
    test_templ(3*np.ones((2,5), dtype=np.uint16))

#--------

def test_02() :
    from ndarray import test_nda_fused_v2
    print(50*'_', '\nTest of templated function test_nda_fused_v2')
    nda = np.arange(0, 10, 1, dtype=np.int32)
    nda.shape = (2,5)
    #print('XXXXXXX type(nda.shape[0]) :', type(nda.shape[0]))

    print('dir(nda):', dir(nda))
    print('nda.strides:', nda.strides)

    print_ndarr(nda, '  nda in : ', first=0, last=10)
    test_nda_fused_v2(nda)
    print_ndarr(nda, '  nda out: ', first=0, last=10)

#--------

def test_03() :
    print(50*'_', '\nTest of py_ctest_vector')
    from ndarray import py_ctest_vector

    nda1 = np.arange(0, 10, 1, dtype=np.double)
    nda2 = np.arange(1, 10, 1, dtype=np.float32)
    nda3 = np.arange(10, 1,-1, dtype=np.int32)
    nda4 = np.ones((5,), dtype=np.double)

    py_ctest_vector(nda1)
    py_ctest_vector(nda2)
    py_ctest_vector(nda3)
    py_ctest_vector(nda4)

    print_ndarr(nda1, '  nda1: ')
    print_ndarr(nda2, '  nda2: ')
    print_ndarr(nda3, '  nda3: ')
    print_ndarr(nda4, '  nda4: ')

#--------

def test_04() :
    print(50*'_', '\nTest of py_ndarray')
    from ndarray import py_ndarray_double
    a = py_ndarray_double()
    a.set_nda(np.ones((2,3), dtype=np.double))

#--------

def test_05() :
    print(50*'_', '\nVoid')

#--------

def test_06() :
    print(50*'_', '\nTest of py_find_edges_v2')
    from ndarray import py_find_edges_v2
    from ex_wf import WF # local import
    #print(WF)
    #wf  = np.arange(0, 20, 1, dtype=np.double)
    wf  = np.array(WF, dtype=np.double)
    pkvals = np.zeros((10,), dtype=np.double)
    pkinds = np.zeros((10,), dtype=np.uint32)
    baseline, threshold, fraction, deadtime, leading_edges = 0, -5, 0.5, 0, True
    npks = py_find_edges_v2(wf, baseline, threshold, fraction, deadtime, leading_edges, pkvals, pkinds)
    print('  npks: %d' % npks)
    print_ndarr(pkvals, '  values : ')
    print_ndarr(pkinds, '  times  : ')

#--------

def usage(tname):
    s = '\nUsage: python psana/psana/hexanode/examples/ex-00-np-to-cpp-ndarray.py <test-number>'
    if tname in ('0',)    : s+='\n 0 - test ALL'
    if tname in ('0','1') : s+='\n 1 - templated function test_nda_fused'
    if tname in ('0','2') : s+='\n 2 - templated function test_nda_fused_v2'
    if tname in ('0','3') : s+='\n 3 - test of py_ctest_vector'
    if tname in ('0','4') : s+='\n 4 - test of py_ndarray'
    if tname in ('0','5') : s+='\n 5 - void'
    if tname in ('0','6') : s+='\n 6 - test of py_find_edges_v2 - uses vectors for IO'
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
    if tname in ('0','5') : test_05()
    if tname in ('0','6') : test_06()
    print('%s' % usage(tname))
    sys.exit(s)

#--------
