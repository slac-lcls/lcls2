#------------------------------
# nosetests -sv psana/psana/tests
# 
#------------------------------

import numpy as np

#------------------------------

def test_pyalgos():
    print('\n%s\n%s' % (50*'_', 'In test_pyalgos'))

#------------------------------

def test_hbins():
    print('In pyalgos.test_hbins')
    from psana.pyalgos.generic.HBins import HBins
    o = HBins((1,6), 5)
    print('  binedges():', o.binedges())
    assert(np.array_equal(o.binedges(), np.array((1,2,3,4,5,6), dtype=np.float)))

#------------------------------

def test_utils():
    print('In pyalgos.test_utils')
    import psana.pyalgos.generic.Utils as gu
    assert(gu.get_enviroment(env='USER') == gu.get_login()) 
    fmt = '%Y-%m-%dT%H:%M:%S%z'
    tsec = 1518640378
    tstamp = '2018-02-14T12:32:58-0800'
    assert(gu.str_tstamp(fmt, tsec) == tstamp)
    assert(gu.time_sec_from_stamp(fmt, tstamp) == tsec)

#------------------------------

def test_entropy():
    #print('In pyalgos.test_entropy')
    from psana.pyalgos.generic.Entropy import unitest_entropy
    unitest_entropy()

#------------------------------

def pyalgos() :
    test_pyalgos()
    test_hbins()
    test_utils()
    test_entropy()

#------------------------------

if __name__ == '__main__':
    pyalgos()
    #print('%s' % 50*'_')

#------------------------------
