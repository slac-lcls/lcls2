
#------------------------------

import numpy as np

#------------------------------

def test_00() :
    print(50*'_', '\nTest of include psana.hexanode')
    from hexanode import fib, met1
    met1()
    print('fib(9):', fib(9))

#------------------------------

def test_templ(nda) :
    from hexanode import test_nda
    #print('%s\n%s' % (50*'_', sys._getframe().f_code.co_name))
    test_nda(nda)
    print('Returned array:\n', nda)

#------------------------------

def test_01() :
    print(50*'_', '\nTest of templated function test_nda')
    test_templ(1*np.ones((2,3), dtype=np.double))
    test_templ(2*np.ones((2,4), dtype=np.int16))
    test_templ(3*np.ones((2,5), dtype=np.uint16))

#------------------------------

def test_02() :
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

def test_templ_xxx(nda) :
    from hexanode import test_nda_xxx
    #print('%s\n%s' % (50*'_', sys._getframe().f_code.co_name))
    test_nda_xxx(nda)
    print('Returned array:\n', nda)

#------------------------------

def test_03() :
    print(50*'_', '\nTest of templated function test_nda_xxx for &nda[0,0]')
    test_templ_xxx(1*np.ones((2,3), dtype=np.double))
    test_templ_xxx(2*np.ones((2,4), dtype=np.int16))
    test_templ_xxx(3*np.ones((2,5), dtype=np.uint16))

#------------------------------

def test_09() :
    import hexanode
    print(50*'_', '\nTest print')

#------------------------------

def usage(tname):
    s = '\nUsage: python psana/psana/hexanode/examples/ex-01-np-to-cpp.py <test-number>'
    if tname == '0' or tname == '0' : s+='\n 0 - met1(), fib(9)'
    if tname == '0' or tname == '1' : s+='\n 1 - templated function test_nda'
    if tname == '0' or tname == '2' : s+='\n 2 - methods test_nda_f8(nda), test_nda_i2(nda), test_nda_u2(nda)'
    if tname == '0' or tname == '3' : s+='\n 3 - templated function test_nda_xxx for &nda[0,0]'
    if tname == '0' or tname == '9' : s+='\n 9 - test import hexanode...'
    return s

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    import numpy as np; global np
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print('%s' % usage(tname))
    print(50*'_', '\nTest %s' % tname)
    if   tname == '0': test_00()
    elif tname == '1': test_01()
    elif tname == '2': test_02()
    elif tname == '3': test_03()
    elif tname == '9': test_09()

    else : sys.exit('Test %s is not implemented' % tname)
    sys.exit('End of Test %s' % tname)

#------------------------------
