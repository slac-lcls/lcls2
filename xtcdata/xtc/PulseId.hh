#ifndef PDS_PULSEID_HH
#define PDS_PULSEID_HH

#include <stdint.h>

namespace XtcData
{
class PulseId
{
public:
    enum { NumPulseIdBits = 56 };

public:
    PulseId();
    PulseId(const PulseId&);
    PulseId(const PulseId&, unsigned control);
    PulseId(uint64_t pulseId, unsigned control = 0);

public:
    uint64_t value()   const; // 929kHz pulse ID
    unsigned control() const; // internal bits for alternate interpretation
    //   of XTC header fields
public:
    PulseId& operator=(const PulseId&);
    bool operator==(const PulseId&) const;
    bool operator!=(const PulseId&) const;
    bool operator>=(const PulseId&) const;
    bool operator<=(const PulseId&) const;
    bool operator<(const PulseId&) const;
    bool operator>(const PulseId&) const;

private:
    uint64_t _value;
};
}

#endif
