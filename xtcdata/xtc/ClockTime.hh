#ifndef XtcData_ClockTime_hh
#define XtcData_ClockTime_hh

#include <stdint.h>
#include <time.h>

namespace XtcData
{
class ClockTime
{
    public:
    ClockTime();
    ClockTime(const ClockTime& t);
    ClockTime(const ::timespec& ts);
    ClockTime(const double sec);
    ClockTime(unsigned sec, unsigned nsec);

    public:
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
    ClockTime& operator=(const ClockTime&);
    bool operator>(const ClockTime&) const;
    bool operator==(const ClockTime&) const;

    private:
    uint32_t _low;
    uint32_t _high;
};
}
#endif
