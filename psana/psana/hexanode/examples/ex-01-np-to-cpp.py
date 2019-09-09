
#------------------------------

import numpy as np
from hexanode import test_nda_f8, test_nda_i2, test_nda_u2

#------------------------------

def test_templ(nda) :
    from hexanode import test_nda
    #print '%s\n%s' % (50*'_', sys._getframe().f_code.co_name)
    test_nda(nda)
    print 'Returned array:\n', nda

#------------------------------

def test_01() :
    print 50*'_', '\nTest of templated function test_nda'
    test_templ(1*np.ones((2,3), dtype=np.double))
    test_templ(2*np.ones((2,4), dtype=np.int16))
    test_templ(3*np.ones((2,5), dtype=np.uint16))

#------------------------------

def test_02() :
    print 50*'_', '\nTest of specialized methods test_nda_f8(nda), test_nda_i2(nda), test_nda_u2(nda)'
    nda = np.ones((2,3), dtype=np.double)
    test_nda_f8(nda)
    print 'Returned array:\n', nda

    nda = 2*np.ones((2,4), dtype=np.int16)
    test_nda_i2(nda)
    print 'Returned array:\n', nda

    nda = 3*np.ones((2,5), dtype=np.uint16)
    test_nda_u2(nda)
    print 'Returned array:\n', nda

#------------------------------

def test_templ_xxx(nda) :
    from hexanode import test_nda_xxx
    #print '%s\n%s' % (50*'_', sys._getframe().f_code.co_name)
    test_nda_xxx(nda)
    print 'Returned array:\n', nda

#------------------------------

def test_03() :
    print 50*'_', '\nTest of templated function test_nda_xxx for &nda[0,0]'
    test_templ_xxx(1*np.ones((2,3), dtype=np.double))
    test_templ_xxx(2*np.ones((2,4), dtype=np.int16))
    test_templ_xxx(3*np.ones((2,5), dtype=np.uint16))

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    import numpy as np; global np
    tname = sys.argv[1] if len(sys.argv) > 1 else '2'
    print 50*'_', '\nTest %s' % tname
    if   tname == '0': test_01(); test_02();
    elif tname == '1': test_01()
    elif tname == '2': test_02()
    elif tname == '3': test_03()

    else : sys.exit('Test %s is not implemented' % tname)
    sys.exit('End of Test %s' % tname)

#------------------------------
