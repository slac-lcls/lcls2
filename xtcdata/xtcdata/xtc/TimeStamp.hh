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

public:
    uint64_t value() const
    {
        return ((uint64_t)_high << 32) | _low;
    }
    unsigned seconds() const
    {
        return _high;
    }
    unsigned nanoseconds() const
    {
        return _low;
    }
    double asDouble() const;
    bool isZero() const;

public:
    TimeStamp& operator=(const TimeStamp&);
    bool operator>(const TimeStamp&) const;
    bool operator==(const TimeStamp&) const;

private:
    uint32_t _low;
    uint32_t _high;
};
}
#endif
