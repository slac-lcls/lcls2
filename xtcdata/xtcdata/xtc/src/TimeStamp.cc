#include "xtcdata/xtc/TimeStamp.hh"
#include <math.h>

XtcData::TimeStamp::TimeStamp()
{
}

XtcData::TimeStamp::TimeStamp(const TimeStamp& t) : _low(t._low), _high(t._high)
{
}

XtcData::TimeStamp::TimeStamp(const timespec& ts) : _low(ts.tv_nsec), _high(ts.tv_sec)
{
}

XtcData::TimeStamp::TimeStamp(unsigned sec, unsigned nsec) : _low(nsec), _high(sec)
{
}

XtcData::TimeStamp::TimeStamp(uint64_t stamp)
{
    _low = stamp&0xffffffff;
    _high = (stamp>>32)&0xffffffff;
}

XtcData::TimeStamp::TimeStamp(double sec)
{
    double intpart;
    double fracpart = modf(sec, &intpart);
    _high = (unsigned)intpart;
    _low = (unsigned)(1.e9 * fracpart + 0.5);
}

bool XtcData::TimeStamp::isZero() const
{
    return _low == 0 && _high == 0;
}

double XtcData::TimeStamp::asDouble() const
{
    return _high + _low / 1.e9;
}

XtcData::TimeStamp& XtcData::TimeStamp::TimeStamp::operator=(const TimeStamp& input)
{
    _low = input._low;
    _high = input._high;
    return *this;
}

bool XtcData::TimeStamp::TimeStamp::operator>(const TimeStamp& t) const
{
    return (_high > t._high) || (_high == t._high && _low > t._low);
}

bool XtcData::TimeStamp::TimeStamp::operator==(const TimeStamp& t) const
{
    return (_high == t._high) && (_low == t._low);
}
