
#------------------------------

import numpy as np

#------------------------------

def test_01() :
    print(50*'_', '\nTest of include psana.hexanode')
    from hexanode import fib, met1
    met1()
    print('fib(9):', fib(9))

#------------------------------

def test_templ(nda) :
    from hexanode import test_nda
    test_nda(nda)
    print('Returned array:\n', nda)

#------------------------------

def test_02() :
    print(50*'_', '\nTest of templated function test_nda')
    test_templ(1*np.ones((2,3), dtype=np.double))
    test_templ(2*np.ones((2,4), dtype=np.int16))
    test_templ(3*np.ones((2,5), dtype=np.uint16))

#------------------------------

def test_03() :
    from hexanode import test_nda_f8, test_nda_i2, test_nda_u2

    print(50*'_', '\nTest of specialized methods test_nda_f8(nda), test_nda_i2(nda), test_nda_u2(nda)')
    nda = np.ones((2,3), dtype=np.double)
    test_nda_f8(nda)
    print('Returned array:\n', nda)

    nda = 2*np.ones((2,4), dtype=np.int16)
    test_nda_i2(nda)
    print('Returned array:\n', nda)

    nda = 3*np.ones((2,5), dtype=np.uint16)
    test_nda_u2(nda)
    print('Returned array:\n', nda)

#------------------------------

def test_09() :
    import hexanode
    print(50*'_', '\nTest print')

#------------------------------

def usage(tname):
    s = '\nUsage: python psana/psana/hexanode/examples/ex-01-np-to-cpp.py <test-number>'
    if tname in ('0',)    : s+='\n 0 - test ALL'
    if tname in ('0','1') : s+='\n 1 - met1(), fib(9)'
    if tname in ('0','2') : s+='\n 2 - templated function test_nda'
    if tname in ('0','3') : s+='\n 3 - methods test_nda_f8(nda), test_nda_i2(nda), test_nda_u2(nda)'
    if tname in ('0','9') : s+='\n 9 - test import hexanode'
    return s

#------------------------------

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
    if tname in ('0','9') : test_09()
    print('%s' % usage(tname))
    sys.exit(s)

#------------------------------
