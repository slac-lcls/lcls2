#------------------------------
"""
Created on 2008-10-08 by Andrei Salnikov
"""
#------------------------------

import sys
import time
import re

#------------------------------

# regular expression to match %f format
_ffmtre = re.compile( "%([.](\\d+))?f" )

# regular expressions for parsing date and time
_DATE_RE = "(\\d{4})(?:-?(\\d{2})(?:-?(\\d{2}))?)?"
_TIME_RE = "(\\d{1,2})(?::?(\\d{2})(?::?(\\d{2})(?:[.](\\d{1,9}))?)?)?"
_TZ_RE = "Z|(?:([-+])(\\d{2})(?::?(\\d{2}))?)"
_dtre = re.compile ( "^" + _DATE_RE + "(?:(?: +|T)(?:" + _TIME_RE + ")?(" + _TZ_RE + ")?)?$" )
  
# time specified as seconds.fractions
_secre = re.compile ( "^S(\\d{0,10})(?:[.](\\d{1,9}))?$" )

def _getNsec ( nsecStr ) :
    """
    Turn string into nanoseconds, strings is everything that
    appears after decimal dot.
    "1"    -> 100000000 ns
    "123"  -> 123000000 ns
    "123456789987654321" -> 123456789ns (truncation, no rounding)
    """
    ndig = min(len(nsecStr),9)

    nsecStr = nsecStr[:ndig] + '0'*(9-ndig)
    return int(nsecStr)

def _cmp_tm( lhs, rhs ) :
    """  compare two tm structs """
    if lhs[0] != rhs[0] : return False ; 
    if lhs[1] != rhs[1] : return False ; 
    if lhs[2] != rhs[2] : return False ; 
    if lhs[3] != rhs[3] : return False ; 
    if lhs[4] != rhs[4] : return False ; 
    if lhs[5] != rhs[5] : return False ; 
    if lhs[8] >= 0 and rhs[8] >= 0 :
        if lhs[8] != rhs[8] : return False ;
    return True ;


#------------------------
# Exported definitions --
#------------------------

def formatTime ( sec, nsec, fmt ):
    """ Convert given time to a string presentation according to
    a given control sequence """

    # replace %f (and its variations) with fractional seconds
    match = _ffmtre.search ( fmt, 0 )
    while match :
        
        # make replacement string
        subsec = ".%09d" % nsec
        
        if match.group(2) :
            # precision is ginen in a format string
            precision = int(match.group(2))
            # bring it into range 1..9
            precision = max ( min ( precision, 9 ), 1 )
            # truncate replacement string
            subsec = subsec[:precision+1]

        # replace %f with this string
        fmt = fmt[:match.start()] + subsec + fmt[match.end():]
        
        # try again
        match = _ffmtre.search ( fmt, match.start() )

    # Python strftime has trouble with %z, we replace it ourselves
    zi = fmt.find("%z")
    while zi >= 0 :
        lt = time.localtime(sec)
        if lt.tm_isdst > 0 and time.daylight:
            utc_offset_minutes = - int(time.altzone/60)
        else:
            utc_offset_minutes = - int(time.timezone/60)
        utc_offset_str = "%+03d%02d" % (utc_offset_minutes/60.0, utc_offset_minutes % 60)
        fmt = fmt[:zi] + utc_offset_str + fmt[zi+2:]
        # try again
        zi = fmt.find("%z")

    # format seconds according to format string
    t = time.localtime ( sec )
    return time.strftime ( fmt, t )

def parseTime ( timeStr ):
    """ Parse the date/time string and return seconds/nanoseconds """

    # try S<sec>.<nsec> format first
    match = _secre.match ( timeStr )
    if match :
        sec = int( match.group(1) )
        nsec = 0
        if match.group(2) : nsec = _getNsec( match.group(2) )
        return ( sec, nsec )

    # next try complex date/time/zone notation
    match = _dtre.match ( timeStr )
    if match :
        
        year = int ( match.group(1) )
        month = int ( match.group(2) or 1 )
        if month < 1 or month > 12 : raise ValueError("parseTime: month value out of range: "+timeStr)
        day = int ( match.group(3) or 1 )
        if day < 1 or day > 31 : raise ValueError("parseTime: day value out of range: "+timeStr)
    
        hour = int ( match.group(4) or 0 )
        if hour > 23 : raise ValueError("parseTime: hour value out of range: "+timeStr)
        minute = int ( match.group(5) or 0 )
        if minute > 59 : raise ValueError("parseTime: minute value out of range: "+timeStr)
        sec = int ( match.group(6) or 0 )
        if sec > 60 : raise ValueError("parseTime: second value out of range: "+timeStr)
        nsec = 0
        if match.group(7) : nsec = _getNsec ( match.group(7) )

        if match.group(8) : 
        
            # timezone offset is given
            tzoffset_min = 0
            if match.group(8) != 'Z' :
                # we have proper offset, calculate offset in minutes, will adjust it later
                tz_hour = int ( match.group(10) )
                tz_min = int ( match.group(11) or 0 )
                if tz_hour > 12 or tz_min > 59 : raise ValueError("parseTime: timezone out of range: "+timeStr)
                tzoffset_min = tz_hour * 60 + tz_min
                if match.group(9) == "-" : tzoffset_min = -tzoffset_min

            # time is in UTC
            isdst = 0
            t = ( year, month, day, hour, minute, sec, -1, -1, isdst )
            sec = mktime_from_utc( t )
            
            # to validate the input convert it back to struct tm
            tval = time.gmtime ( sec )
            if not _cmp_tm( t, tval ) : raise ValueError( "parseTime: input time validation failed: "+timeStr ) ;
            
            # adjust for timezone
            sec -= tzoffset_min * 60 ;
            
        else :
            
            # No timezone specified, we should assume the time is in the local timezone.
            # Let it guess the daylight saving time status.
            isdst = -1
            t = ( year, month, day, hour, minute, sec, -1, -1, isdst )
            sec = time.mktime( t )
          
            # to validate the input convert it back to struct tm
            tval = time.localtime ( sec )
            if not _cmp_tm( t, tval ) : raise ValueError( "parseTime: input time validation failed: "+timeStr ) ;

        return ( sec, nsec )

    # no match found
    raise ValueError("parseTime: failed to parse string: "+timeStr)


def mktime_from_utc (t) :
    """
    /* Converts struct tm to time_t, assuming the data in tm is UTC rather
       than local timezone.
    
       mktime is similar but assumes struct tm, also known as the
       "broken-down" form of time, is in local time zone.  mktime_from_utc
       uses mktime to make the conversion understanding that an offset
       will be introduced by the local time assumption.
    
       mktime_from_utc then measures the introduced offset by applying
       gmtime to the initial result and applying mktime to the resulting
       "broken-down" form.  The difference between the two mktime results
       is the measured offset which is then subtracted from the initial
       mktime result to yield a calendar time which is the value returned.
    
       tm_isdst in struct tm is set to 0 to force mktime to introduce a
       consistent offset (the non DST offset) since tm and tm+o might be
       on opposite sides of a DST change.
    
       Some implementations of mktime return -1 for the nonexistent
       localtime hour at the beginning of DST.  In this event, use
       mktime(tm - 1hr) + 3600.
    
       Schematically
         mktime(tm)   --> t+o
         gmtime(t+o)  --> tm+o
         mktime(tm+o) --> t+2o
         t+o - (t+2o - t+o) = t
    
       Note that glibc contains a function of the same purpose named
       `timegm' (reverse of gmtime).  But obviously, it is not universally
       available, and unfortunately it is not straightforwardly
       extractable for use here.  Perhaps configure should detect timegm
       and use it where available.
    
       Contributed by Roger Beeman <beeman@cisco.com>, with the help of
       Mark Baushke <mdb@cisco.com> and the rest of the Gurus at CISCO.
       Further improved by Roger with assistance from Edward J. Sabol
       based on input by Jamie Zawinski.  */
    """

    try :
        tl = time.mktime (t);
    except :
        t = ( t[0], t[1], t[2], t[3]-1, t[4], t[5], t[6], t[7], t[8] )
        tl = time.mktime (t);
        tl += 3600;

    tg = time.gmtime (tl);
    tg = ( tg[0], tg[1], tg[2], tg[3], tg[4], tg[5], tg[6], tg[7], 0 )
    try :
        tb = time.mktime (tg);
    except :
        tg = ( tg[0], tg[1], tg[2], tg[3]-1, tg[4], tg[5], tg[6], tg[7], 0 )
        tb = time.mktime (tg);
        tb += 3600;
    return (tl - (tb - tl));

#------------------------------

if __name__ == "__main__" :
    sys.exit ( "Module is not supposed to be run as main module" )

#------------------------------
