#!/usr/bin/env python
#------------------------------

import psalg_ext

#------------------------------

def test01():
    import psalg_ext
    print('test import psalg_ext')

#------------------------------

def test02():
    import psalg_ext
    import numpy as np
    from psana.pyalgos.generic.NDArrUtils import print_ndarr
    print('test numpy.array')
    
    af8 = np.ones((5,5), dtype=np.float64)
    print_ndarr(af8, 'input array ')
    #psalg_ext.test_nda_f8(af8)
    #psalg_ext.test_nda_v1(af8)
    #print_ndarr(af8, 'output array')

    ai2 = np.ones((6,3), dtype=np.int16)
    print_ndarr(ai2, 'input array ')
    #psalg_ext.test_nda_i2(ai2)
    #psalg_ext.test_nda_v1(ai2)
    #print_ndarr(ai2, 'output array')

#------------------------------
#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print(50*'_', '\nTest %s:' % tname)
    if   tname == '1' : test01()
    elif tname == '2' : test02()
    else : sys.exit('Test %s is not implemented' % tname)
    sys.exit('End of test %s' % tname)

#------------------------------
