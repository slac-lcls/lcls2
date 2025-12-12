#include "xtcdata/xtc/TimeStamp.hh"
#include <math.h>

XtcData::TimeStamp::TimeStamp(double sec)
{
    double intpart;
    double fracpart = modf(sec, &intpart);
    _high = (unsigned)intpart;
    _low = (unsigned)(1.e9 * fracpart + 0.5);
}
