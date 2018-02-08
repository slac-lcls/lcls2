#!/usr/bin/env python
#------------------------------

import psalgos

#------------------------------

def test01():
    print 'call pure python'

#------------------------------

def test02():
    print 'call psalgos.fib(90)'
    print psalgos.fib(90)

#------------------------------

#------------------------------

def test03():
    import numpy as np
    from pyimgalgos.GlobalUtils import print_ndarr
    print 'test numpy.array'
    
    af8 = np.ones((5,5), dtype=np.float64)
    print_ndarr(af8, 'input array ')
    #psalgos.test_nda_f8(af8)
    psalgos.test_nda_v1(af8)
    print_ndarr(af8, 'output array')

    ai2 = np.ones((6,3), dtype=np.int16)
    print_ndarr(ai2, 'input array ')
    #psalgos.test_nda_i2(ai2)
    psalgos.test_nda_v1(ai2)
    print_ndarr(ai2, 'output array')

#------------------------------
#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print 50*'_', '\nTest %s:' % tname
    if   tname == '1' : test01()
    elif tname == '2' : test02()
    elif tname == '3' : test03()
    else : sys.exit('Test %s is not implemented' % tname)
    sys.exit('End of test %s' % tname)

#------------------------------
