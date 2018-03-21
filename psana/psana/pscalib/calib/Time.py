#------------------------------
"""
Created on 2008-10-08 by Andrei Salnikov
"""
#------------------------------
import sys
import time

import psana.pscalib.calib.TimeFormat as TimeFormat
#------------------------------

class Time() :
    """Common time class. 
    
    Counts time since the standard UNIX epoch. Provides nanosecond precision.
    """
    def __init__ ( self, sec = None, nsec = 0 ) :
        """Constructor.

        - sec    seconds since epoch
        - nsec   nanoseconds
        """
        if nsec < 0 or nsec > 999999999 : raise ValueError("nanoseconds value out of range")
        self._sec = sec ;
        self._nsec = nsec ;

    #-------------------
    #  Public methods --
    #-------------------

    def sec(self) :
        return self._sec

    def nsec(self) :
        return self._nsec

    def isValid(self) :
        return self._sec is not None

    def to64(self):
        """ Pack time into a 64-bit number. """
        if self._sec is None : raise ValueError("Time.to64: converting invalid object")
        return self._sec*1000000000 + self._nsec

    def toString(self, fmt="%F %T%f%z" ):
        """ Format time according to format string """
        if self._sec is None : raise ValueError("Time.toString: converting invalid object")
        return TimeFormat.formatTime( self._sec, self._nsec, fmt )

    def __str__ ( self ):
        """ Convert to a string """
        return self.toString()

    def __repr__ ( self ):
        """ Convert to a string """
        if self._sec is None : return "Time()"
        return "<Time:%s>" % self.toString("S%s%f")

    def __cmp__ ( self, other ):
        """ IS NOT USED IN PYTHON 3 !!! compare two Time objects """
        if not isinstance(other,Time) : raise TypeError ( "Time.__cmp__: comparing to unknown type" )
        if self._sec is None or other._sec is None : raise ValueError ( "Time.__cmp__: comparing invalid times" )
        return cmp ( ( self._sec,self._nsec ), ( other._sec,other._nsec ) )

    def _is_valid_to_compare(self, other):
        if not isinstance(other,Time) : raise TypeError ( "Time.__cmp__: comparing to unknown type" )
        if self._sec is None or other._sec is None : raise ValueError ( "Time.__cmp__: comparing invalid times" )

    def __eq__(self, other):
        self._is_valid_to_compare(other)
        return ((self._sec,self._nsec) == (other._sec,other._nsec))

    def __ne__(self, other):
        self._is_valid_to_compare(other)
        return ((self._sec,self._nsec) != (other._sec,other._nsec))
 
    def __lt__(self, other):
        self._is_valid_to_compare(other)
        return ((self._sec,self._nsec) < (other._sec,other._nsec))

    def __le__(self, other):
        self._is_valid_to_compare(other)
        return ((self._sec,self._nsec) <= (other._sec,other._nsec))

    def __gt__(self, other):
        self._is_valid_to_compare(other)
        return ((self._sec,self._nsec) > (other._sec,other._nsec))

    def __ge__(self, other):
        self._is_valid_to_compare(other)
        return ((self._sec,self._nsec) >= (other._sec,other._nsec))

    def __hash__ ( self ):
        """ calculate hash value for use in dictionaries, returned hash value 
        should be 32-bit integer """
        if self._sec is None : raise ValueError ( "Time.__hash__: invalid time value" )
        return hash( (self._sec, self._nsec) )

    #--------------------------------
    #  Static/class public methods --
    #--------------------------------

    @staticmethod
    def now():
        """ Get current time, resolution can be lower than 1ns """
        t = time.time()
        sec = int(t)
        nsec = int( (t-sec) * 1e9 )
        return Time( sec, nsec )

    @staticmethod
    def from64( packed ):
        """ Unpack 64-bit time into time object """
        sec, nsec = divmod( packed, 1000000000 )
        return Time( sec, nsec )

    @staticmethod
    def parse( s ):
        """ Convert string presentation into time object """
        sec, nsec = TimeFormat.parseTime( s )
        return Time( sec, nsec )
    
#------------------------------

if __name__ == "__main__" :
    sys.exit ( "Module is not supposed to be run as main module" )

#------------------------------
