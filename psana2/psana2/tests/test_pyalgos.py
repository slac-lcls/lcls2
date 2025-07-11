
""" pytest psana/psana/tests OR pytest psana/psana/test_pyalgos.py """

import numpy as np

def test_pyalgos():
    print('\n%s\n%s' % (50*'_', 'In test_pyalgos'))

def test_hbins():
    print('In pyalgos.test_hbins')
    from psana2.pyalgos.generic.HBins import HBins
    o = HBins((1,6), 5)
    print('  binedges():', o.binedges())
    assert(np.array_equal(o.binedges(), np.array((1,2,3,4,5,6), dtype=float)))

def test_entropy():
    #print('In pyalgos.test_entropy')
    from psana2.pyalgos.generic.Entropy import unitest_entropy
    unitest_entropy()

def pyalgos() :
    test_pyalgos()
    test_hbins()
    #test_utils()
    test_entropy()

if __name__ == '__main__':
    pyalgos()
    #print('%s' % 50*'_')

# EOF
