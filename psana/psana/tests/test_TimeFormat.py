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

class TestTimeFormat(unittest.TestCase) :

    def test_1(self):
        # test for %f specifier 
  
        t0 = Time(0,123456789) ;
  
        tstr = t0.toString("%f") ;
        self.assertEqual( tstr, ".123456789" ) ;

        tstr = t0.toString("%.1f") ;
        self.assertEqual( tstr, ".1" ) ;
        tstr = t0.toString("%.2f") ;
        self.assertEqual( tstr, ".12" ) ;
        tstr = t0.toString("%.3f") ;
        self.assertEqual( tstr, ".123" ) ;
        tstr = t0.toString("%.4f") ;
        self.assertEqual( tstr, ".1234" ) ;
        tstr = t0.toString("%.5f") ;
        self.assertEqual( tstr, ".12345" ) ;
        tstr = t0.toString("%.6f") ;
        self.assertEqual( tstr, ".123456" ) ;
        tstr = t0.toString("%.7f") ;
        self.assertEqual( tstr, ".1234567" ) ;
        tstr = t0.toString("%.8f") ;
        self.assertEqual( tstr, ".12345678" ) ;
        tstr = t0.toString("%.9f") ;
        self.assertEqual( tstr, ".123456789" ) ;
  
        tstr = t0.toString("%.0f") ;
        self.assertEqual( tstr, ".1" ) ;
        tstr = t0.toString("%.10f") ;
        self.assertEqual( tstr, ".123456789" ) ;
        tstr = t0.toString("%.128f") ;
        self.assertEqual( tstr, ".123456789" ) ;

    def test_2(self):

        # test for date specifiers, assume this test runs in US, -0800 timezone
  
        t0 = Time (0,0) ;
  
        tstr = t0.toString("%Y") ;
        self.assertEqual( tstr, "1969" ) ;
        tstr = t0.toString("%Y-%m") ;
        self.assertEqual( tstr, "1969-12" ) ;
        tstr = t0.toString("%Y-%m-%d") ;
        self.assertEqual( tstr, "1969-12-31" ) ;
        tstr = t0.toString("%F") ;
        self.assertEqual( tstr, "1969-12-31" ) ;
        tstr = t0.toString("%Y-%j") ;
        self.assertEqual( tstr, "1969-365" ) ;

    def test_3(self):
        # test for time specifiers, assume this test runs in US, -0800 timezone
  
        t0 = Time (0,0) ;
  
        tstr = t0.toString("%H:%M:%S") ;
        self.assertEqual( tstr, "16:00:00" ) ;
        tstr = t0.toString("%T") ;
        self.assertEqual( tstr, "16:00:00" ) ;
        tstr = t0.toString("%T%z") ;
        self.assertEqual( tstr, "16:00:00-0800" ) ;

    def test_4(self):
        # test for default time formatting, assume this test runs in US, -0800 timezone
  
        t0 = Time (0,0) ;
  
        tstr = t0.toString() ;
        self.assertEqual( tstr, "1969-12-31 16:00:00.000000000-0800" ) ;
        tstr = str(t0) ;
        self.assertEqual( tstr, "1969-12-31 16:00:00.000000000-0800" ) ;

        t1 = Time (1234567890,123456789) ;
  
        tstr = t1.toString() ;
        self.assertEqual( tstr, "2009-02-13 15:31:30.123456789-0800" ) ;
        tstr = str(t1) ;
        self.assertEqual( tstr, "2009-02-13 15:31:30.123456789-0800" ) ;

#------------------------------

if __name__ == "__main__":
    unittest.main()

#------------------------------
