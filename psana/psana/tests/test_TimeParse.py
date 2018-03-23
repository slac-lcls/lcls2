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

class TestTimeParse(unittest.TestCase) :

    def test_1 (self):
        # test for S<seconds>[.<fractions>] syntax
  
        # test illegal syntax
        self.assertRaises( Exception, lambda : Time.parse("S 1") ) ;
        self.assertRaises( Exception, lambda : Time.parse(" S1") ) ;
        self.assertRaises( Exception, lambda : Time.parse("S1 ") ) ;
        self.assertRaises( Exception, lambda : Time.parse("S12345678900") ) ;
        self.assertRaises( Exception, lambda : Time.parse("S-1") ) ;
        self.assertRaises( Exception, lambda : Time.parse("S1.") ) ;
        self.assertRaises( Exception, lambda : Time.parse("S0.1234567890") ) ;
  
        # test legal syntax, all variations
        t0 = Time.parse("S1234567890") ;
        self.assertEqual ( t0, Time ( 1234567890, 0 ) ) ;
        t0 = Time.parse("S1234567890.1") ;
        self.assertEqual ( t0, Time ( 1234567890, 100000000 ) ) ;
        t0 = Time.parse("S1234567890.123456789") ;
        self.assertEqual ( t0, Time ( 1234567890, 123456789 ) ) ;
        t0 = Time.parse("S0.000000001") ;
        self.assertEqual ( t0, Time ( 0, 1 ) ) ;
        t0 = Time.parse("S0") ;
        self.assertEqual ( t0, Time ( 0, 0 ) ) ;

    def test_2(self):

        # epoch time in different formats
        t0 = Time.parse("1970-01-01 00:00:00Z") ;
        self.assertEqual ( t0, Time ( 0, 0 ) ) ;
        t0 = Time.parse("1970-01-01 00:00:00-00") ;
        self.assertEqual ( t0, Time ( 0, 0 ) ) ;
        t0 = Time.parse("1970-01-01 00:00:00+00:00") ;
        self.assertEqual ( t0, Time ( 0, 0 ) ) ;
        t0 = Time.parse("1970-01-01 00:00:00-0000") ;
        self.assertEqual ( t0, Time ( 0, 0 ) ) ;
        t0 = Time.parse("19700101 000000Z") ;
        self.assertEqual ( t0, Time ( 0, 0 ) ) ;
        t0 = Time.parse("19700101        000000Z") ;
        self.assertEqual ( t0, Time ( 0, 0 ) ) ;
        t0 = Time.parse("19700101T000000Z") ;
        self.assertEqual ( t0, Time ( 0, 0 ) ) ;
        t0 = Time.parse("19700101T010000+01") ;
        self.assertEqual ( t0, Time ( 0, 0 ) ) ;
        t0 = Time.parse("19700101T010000+0100") ;
        self.assertEqual ( t0, Time ( 0, 0 ) ) ;
        t0 = Time.parse("1970-01-01 01:00:00+01:00") ;
        self.assertEqual ( t0, Time ( 0, 0 ) ) ;

    def test_3(self):
        # test for illegal date-time specifier
  
        self.assertRaises( Exception, lambda : Time.parse("1970-02-03 10:11:12.1234567890123456789") ) ;
        self.assertRaises( Exception, lambda : Time.parse("1999-99-03") ) ;
        self.assertRaises( Exception, lambda : Time.parse("1999-01-99") ) ;
        self.assertRaises( Exception, lambda : Time.parse("1999-01-01 25:00:00") ) ;
        self.assertRaises( Exception, lambda : Time.parse("1999-01-01 00:60:00") ) ;
        self.assertRaises( Exception, lambda : Time.parse("1999-01-01 00:00:61") ) ;
        self.assertRaises( Exception, lambda : Time.parse("1999-02-30 00:00:00") ) ;

    def test_4(self):
        # test for timezones
  
        t0 = Time.parse("197002 Z") ;
        self.assertEqual (t0, Time (31 * 24 * 3600, 0)) ;
        t0 = Time.parse("19700102 Z") ;
        self.assertEqual (t0, Time (24 * 3600, 0)) ;
        t0 = Time.parse("1970-01-01 -08") ;
        self.assertEqual (t0, Time (8 * 3600, 0)) ;
        t0 = Time.parse("19700102T000000Z") ;
        self.assertEqual (t0, Time (24 * 3600, 0)) ;
        t0 = Time.parse("19700102T000000+0100") ;
        self.assertEqual (t0, Time (23 * 3600, 0)) ;
        t0 = Time.parse("19700102T000000+1000") ;
        self.assertEqual (t0, Time (14 * 3600, 0)) ;
        t0 = Time.parse("19700102T000000-1000") ;
        self.assertEqual (t0, Time (34 * 3600, 0)) ;
        t0 = Time.parse("19700102T000000-0140") ;
        self.assertEqual (t0, Time (24 * 3600 + 100 * 60, 0)) ;
        t0 = Time.parse("19700102T000000+0140") ;
        self.assertEqual (t0, Time (24 * 3600 - 100 * 60, 0)) ;
        t0 = Time.parse("19700102T000000-01:40") ;
        self.assertEqual (t0, Time (24 * 3600 + 100 * 60, 0)) ;
        t0 = Time.parse("19700102T000000+01:40") ;
        self.assertEqual (t0, Time (24 * 3600 - 100 * 60, 0)) ;


    def test_5(self):
        # test for local timezone
  
        Time.parse("20010101") ;
        Time.parse("20010101 01:01:01.01") ;
        self.assertRaises( Exception, lambda : Time.parse("20010431 01:01:01.01") ) ;

#------------------------------

if __name__ == "__main__":
    unittest.main()

#------------------------------
