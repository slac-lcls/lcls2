#!/usr/bin/env python

from psana2.pyalgos.generic.HBins import * #HBins

def test_bin_indexes(o, vals, edgemode=0, cmt=''):
    print('%s\n%s, edgemode=%d:' % (80*'_', cmt, edgemode))
    print('nbins = %d' % o.nbins())
    print('binedges',    o.binedges())
    print('equalbins',   o.equalbins())

    print('Test of o.bin_index:')
    for v in vals: print('value=%5.1f index=%2d' % (v, o.bin_index(v, edgemode)))

    print('Test of o.bin_indexes:')
    inds = o.bin_indexes(vals, edgemode)
    for v,i in zip(vals,inds): print('value=%5.1f index=%2d' % (v, i))


def test(o, cmt=''):

    print('%s\n%s\n' % (80*'_', cmt))

    o.print_attrs_and_methods()
    o.print_attrs_defined()
    print('nbins = %d' %   o.nbins())
    print('limits',        o.limits())
    print('binedges',      o.binedges())
    print('binedgesleft',  o.binedgesleft())
    print('binedgesright', o.binedgesright())
    print('bincenters',    o.bincenters())
    print('binwidth',      o.binwidth())
    print('halfbinw',      o.halfbinw())
    print('strrange',      o.strrange())
    print('equalbins',     o.equalbins())
    o.print_attrs_defined()
    print('%s' % (80*'_'))


def test_bin_data(o, cmt=''):
    print('%s\n%s' % (80*'_', cmt))
    data = np.arange(o.nbins())
    o.set_bin_data(data, dtype=np.int32)
    data_ret = o.bin_data(dtype=np.int32)
    print('data saved   :', data)
    print('data retrieved:', data_ret)


if __name__ == "__main__":

    o1 = HBins((1,6), 5);     test(o1, 'Test HBins for EQUAL BINS')
    o2 = HBins((1, 2, 4, 8)); test(o2, 'Test HBins for VARIABLE BINS')

    try: o = HBins((1,6), 5.5)
    except Exception as e: print('Test Exception non-int nbins:', e)

    try: o = HBins((1,6), -5)
    except Exception as e: print('Test Exception nbins<1:', e)

    try: o = HBins((1,6), 0)
    except Exception as e: print('Test Exception nbins<1:', e)

    try: o = HBins((1,6,3))
    except Exception as e: print('Test Exception non-monotonic edges:', e)

    try: o = HBins((3,6,1))
    except Exception as e: print('Test Exception non-monotonic edges:', e)

    try: o = HBins((3,2,2,1))
    except Exception as e: print('Test Exception non-monotonic edges:', e)

    try: o = HBins((3,'s',1))
    except Exception as e: print('Test Exception wrong type value in edges:', e)

    try: o = HBins(3)
    except Exception as e: print('Test Exception not-sequence in edges:', e)

    try: o = HBins((3,))
    except Exception as e: print('Test Exception sequence<2 in edges:', e)

    vals=(-3, 0, 1, 1.5, 2, 3, 4, 5, 6, 8, 10)
    test_bin_indexes(o1, vals, edgemode=0, cmt='Test for EQUAL BINS')
    test_bin_indexes(o1, vals, edgemode=1, cmt='Test for EQUAL BINS')
    test_bin_indexes(o2, vals, edgemode=0, cmt='Test for VARIABLE BINS')
    test_bin_indexes(o2, vals, edgemode=1, cmt='Test for VARIABLE BINS')

    test_bin_data(o1, cmt='Test set_bin_data and bin_data methods')

# EOF
