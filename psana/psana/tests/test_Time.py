#------------------------------

"""Unit test application for psana.pscalib.calib.Time (Lusi.Time) class.

This software was developed for the LUSI project.
If you use all or part of it, please give an appropriate acknowledgement.

Andrei Salnikov
"""
#------------------------------

import sys
import os
import unittest

from psana.pscalib.calib.Time import *

#------------------------------

class TestTime(unittest.TestCase) :

    def setUp(self) :
        pass
    
    def tearDown(self) :
        pass

    def testValid(self):
        t0 = Time()
        self.assertTrue( not t0.isValid() )
        t0 = Time(123)
        self.assertTrue( t0.isValid() )
        t0 = Time(12345,12345)
        self.assertTrue( t0.isValid() )

    def testFieldValues(self):
        t0 = Time( 1234567, 123456 )
        self.assertTrue( t0.isValid() )
        self.assertEqual( t0.sec(), 1234567 )
        self.assertEqual( t0.nsec(), 123456 )

    def test_cmp_1(self):
        t0 = Time( 1234567, 123456 )
        t1 = Time( 1234567, 123456 )
        self.assertTrue( t0 == t1 )
        self.assertTrue( not ( t0 != t1 ) )
        self.assertTrue( not ( t0 < t1 ) )
        self.assertTrue( t0 <= t1 )
        self.assertTrue( not ( t0 > t1 ) )
        self.assertTrue( t0 >= t1 )

    def test_cmp_2(self):
        t0 = Time( 1234567, 0 )
        t1 = Time( 1234567, 123456 )
        self.assertTrue( not ( t0 == t1 ) )
        self.assertTrue( t0 != t1 )
        self.assertTrue( t0 < t1 )
        self.assertTrue( t0 <= t1 )
        self.assertTrue( not ( t0 > t1 ) )
        self.assertTrue( not ( t0 >= t1 ) )

    def test_cmp_3(self):
        t0 = Time( 123, 123 )
        t1 = Time( 1234567, 123456 )
        self.assertTrue( not ( t0 == t1 ) )
        self.assertTrue( t0 != t1 )
        self.assertTrue( t0 < t1 )
        self.assertTrue( t0 <= t1 )
        self.assertTrue( not ( t0 > t1 ) )
        self.assertTrue( not ( t0 >= t1 ) )

    def test_cmp_4(self):
        t0 = Time( 123, 123456 )
        t1 = Time( 1234567, 123456 )
        self.assertTrue( not ( t0 == t1 ) )
        self.assertTrue( t0 != t1 )
        self.assertTrue( t0 < t1 )
        self.assertTrue( t0 <= t1 )
        self.assertTrue( not ( t0 > t1 ) )
        self.assertTrue( not ( t0 >= t1 ) )

    def test_cmp_5(self):
        t0 = Time( 1234567, 123456 )
        t1 = Time( 123, 123 )
        self.assertTrue( not ( t0 == t1 ) )
        self.assertTrue( t0 != t1 )
        self.assertTrue( not ( t0 < t1 ) )
        self.assertTrue( not ( t0 <= t1 ) )
        self.assertTrue( t0 > t1 )
        self.assertTrue( t0 >= t1 )

    def test_cmp_6(self):
        t0 = Time()
        t1 = Time( 123, 123 )
        self.assertRaises( Exception, lambda : t0 == t1 )
        self.assertRaises( Exception, lambda : t0 != t1 )
        self.assertRaises( Exception, lambda : t0 < t1 )
        self.assertRaises( Exception, lambda : t0 <= t1 )
        self.assertRaises( Exception, lambda : t0 > t1 )
        self.assertRaises( Exception, lambda : t0 >= t1 )

    def test_cmp_7(self):
        t0 = Time( 123, 123 ) 
        t1 = Time()
        self.assertRaises( Exception, lambda : t0 == t1 )
        self.assertRaises( Exception, lambda : t0 != t1 )
        self.assertRaises( Exception, lambda : t0 < t1 )
        self.assertRaises( Exception, lambda : t0 <= t1 )
        self.assertRaises( Exception, lambda : t0 > t1 )
        self.assertRaises( Exception, lambda : t0 >= t1 )

    def test_cmp_8(self):
        t0 = Time()
        t1 = Time()
        self.assertRaises( Exception, lambda : t0 == t1 )
        self.assertRaises( Exception, lambda : t0 != t1 )
        self.assertRaises( Exception, lambda : t0 < t1 )
        self.assertRaises( Exception, lambda : t0 <= t1 )
        self.assertRaises( Exception, lambda : t0 > t1 )
        self.assertRaises( Exception, lambda : t0 >= t1 )

    def test_hash(self):

        d = {}
        d[Time(1)] = 1000
        d[Time(2)] = 10000
        d[Time(1000)] = 3

        self.assertEqual( len(d), 3 )
        self.assertEqual( d[Time(1)], 1000 )
        self.assertEqual( d[Time(2)], 10000 )
        self.assertEqual( d[Time(1000)], 3 )
        self.assertEqual( d.get(Time(5),5), 5 )
        self.assertRaises( Exception, lambda : d.get(Time(),None) )

#------------------------------

if __name__ == "__main__":
    unittest.main()

#------------------------------
