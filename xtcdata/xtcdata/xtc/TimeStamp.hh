/**
 * @file
 * @brief Declares XtcData::TimeStamp, a 64-bit time stored as 32-bit seconds and 32-bit nanoseconds words.
 */
#ifndef XtcData_TimeStamp_hh
#define XtcData_TimeStamp_hh

#include <stdint.h>
#include <time.h>

namespace XtcData
{
/** Time stored as two 32-bit words: seconds (high word) and nanoseconds (low word). value() returns them packed as (seconds << 32) | nanoseconds. */
class TimeStamp
{
public:
    /** Default constructor; both words are left uninitialized. */
    TimeStamp();
    /** Copy both words of t. */
    TimeStamp(const TimeStamp& t);
    /** Set seconds from ts.tv_sec and nanoseconds from ts.tv_nsec. */
    TimeStamp(const ::timespec& ts);
    /** Split sec into whole seconds and a fraction; nanoseconds = (unsigned)(1e9 * fraction + 0.5). */
    TimeStamp(double sec);
    /** Set seconds to sec and nanoseconds to nsec (not normalized). */
    TimeStamp(unsigned sec, unsigned nsec);
    /** Set nanoseconds to bits 31:0 of stamp and seconds to bits 63:32 (the inverse of value()). */
    TimeStamp(uint64_t stamp);

public:
    /** Return (seconds << 32) | nanoseconds. */
    uint64_t   value() const;
    /** Return the seconds (high) word. */
    unsigned   seconds() const;
    /** Return the nanoseconds (low) word. */
    unsigned   nanoseconds() const;
    /** Return seconds + nanoseconds / 1e9. */
    double     asDouble() const;
    /** Return true if both words are 0. */
    bool       isZero() const;
    /** Return seconds * 1000000000 + nanoseconds. */
    uint64_t   to_ns() const;
    /**
     * Set seconds to nsec / 1000000000 (truncated to 32 bits) and nanoseconds to nsec % 1000000000.
     * @return *this.
     */
    TimeStamp& from_ns(uint64_t nsec);
public:
    /**
     * Copy both words.
     * @return *this.
     */
    TimeStamp& operator=(const TimeStamp&);
    /** Return true if this time is later: compares seconds, then nanoseconds. */
    bool       operator>(const TimeStamp&) const;
    /** Return true if both words are equal. */
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
XtcData::TimeStamp& XtcData::TimeStamp::from_ns(uint64_t nsec)
{
    _low  = nsec % 1000000000;
    _high = nsec / 1000000000;

    return *this;
}

inline
uint64_t XtcData::TimeStamp::to_ns() const
{
    return ((uint64_t)_high * 1000000000) + _low;
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
