#ifndef PDS_TIMESTAMP_HH
#define PDS_TIMESTAMP_HH

#include <stdint.h>

namespace XtcData
{
class TimeStamp
{
public:
    enum { NumPulseIdBits = 56 };

public:
    TimeStamp();
    TimeStamp(const TimeStamp&);
    TimeStamp(const TimeStamp&, unsigned control);
    TimeStamp(uint64_t pulseId, unsigned control = 0);

public:
    unsigned pulseId() const; // 929kHz pulse ID
    unsigned control() const; // internal bits for alternate interpretation
    //   of XTC header fields
public:
    TimeStamp& operator=(const TimeStamp&);
    bool operator==(const TimeStamp&) const;
    bool operator!=(const TimeStamp&) const;
    bool operator>=(const TimeStamp&) const;
    bool operator<=(const TimeStamp&) const;
    bool operator<(const TimeStamp&) const;
    bool operator>(const TimeStamp&) const;

private:
    uint64_t _value;
};
}

#endif
