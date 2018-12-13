#ifndef XtcData_TimeStamp_hh
#define XtcData_TimeStamp_hh

#include <stdint.h>
#include <time.h>

namespace XtcData
{
class TimeStamp
{
public:
    TimeStamp();
    TimeStamp(const TimeStamp& t);
    TimeStamp(const ::timespec& ts);
    TimeStamp(const double sec);
    TimeStamp(unsigned sec, unsigned nsec);
    TimeStamp(uint64_t stamp);

public:
    uint64_t value() const;
    unsigned seconds() const;
    unsigned nanoseconds() const;
    double   asDouble() const;
    bool     isZero() const;

public:
    TimeStamp& operator=(const TimeStamp&);
    bool       operator>(const TimeStamp&) const;
    bool       operator==(const TimeStamp&) const;

private:
    uint32_t _low;
    uint32_t _high;
};


inline
XtcData::TimeStamp::TimeStamp()
{
}

inline
XtcData::TimeStamp::TimeStamp(const TimeStamp& t) : _low(t._low), _high(t._high)
{
}

inline
XtcData::TimeStamp::TimeStamp(const timespec& ts) : _low(ts.tv_nsec), _high(ts.tv_sec)
{
}

inline
XtcData::TimeStamp::TimeStamp(unsigned sec, unsigned nsec) : _low(nsec), _high(sec)
{
}

inline
XtcData::TimeStamp::TimeStamp(uint64_t stamp)
{
    _low  =  stamp        & 0xffffffff;
    _high = (stamp >> 32) & 0xffffffff;
}

inline
uint64_t XtcData::TimeStamp::value() const
{
    return ((uint64_t)_high << 32) | _low;
}

inline
unsigned XtcData::TimeStamp::seconds() const
{
    return _high;
}

inline
unsigned XtcData::TimeStamp::nanoseconds() const
{
    return _low;
}

inline
bool XtcData::TimeStamp::isZero() const
{
    return _low == 0 && _high == 0;
}

inline
double XtcData::TimeStamp::asDouble() const
{
    return _high + _low / 1.e9;
}

inline
XtcData::TimeStamp& XtcData::TimeStamp::TimeStamp::operator=(const TimeStamp& input)
{
    _low  = input._low;
    _high = input._high;
    return *this;
}

inline
bool XtcData::TimeStamp::TimeStamp::operator>(const TimeStamp& t) const
{
    return (_high > t._high) || (_high == t._high && _low > t._low);
}

inline
bool XtcData::TimeStamp::TimeStamp::operator==(const TimeStamp& t) const
{
    return (_high == t._high) && (_low == t._low);
}
}
#endif
