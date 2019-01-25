#ifndef PDS_PULSEID_HH
#define PDS_PULSEID_HH

#include <stdint.h>

namespace XtcData
{
class PulseId
{
public:
    enum { NumPulseIdBits = 56 };

    /* bit field access enums
    *       v is the index of the rightmost bit
    *       k is the number bits in the field
    *       m is the mask, right justified
    *       s is the mask shifted into place
    */
    enum { v_pulseId = 0, k_pulseId = 56 };
    enum { v_cntrl = 56, k_cntrl = 8 };

    static const uint64_t m_pulseId = ((1ULL << k_pulseId) - 1);
    static const uint64_t s_pulseId = (m_pulseId << v_pulseId);
    static const uint64_t m_cntrl = ((1ULL << k_cntrl) - 1);
    static const uint64_t s_cntrl = (m_cntrl << v_cntrl);

public:
    PulseId();
    PulseId(const PulseId&);
    PulseId(const PulseId&, unsigned control);
    PulseId(uint64_t pulseId, unsigned control = 0);

public:
    uint64_t value()   const; // 929kHz pulse ID
    unsigned control() const; // internal bits for alternate interpretation
                              // of XTC header fields
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


inline
XtcData::PulseId::PulseId()
  : _value(0)
{
}

inline
XtcData::PulseId::PulseId(const XtcData::PulseId& input)
  : _value(input._value)
{
}

inline
XtcData::PulseId::PulseId(const XtcData::PulseId& input, unsigned control)
  : _value((input._value & s_pulseId) | ((control & m_cntrl) << v_cntrl))
{
}

inline
XtcData::PulseId::PulseId(uint64_t pulseId, unsigned control)
  : _value((pulseId & s_pulseId) | ((control & m_cntrl) << v_cntrl))
{
}

inline
uint64_t XtcData::PulseId::value() const
{
    return (_value & s_pulseId) >> v_pulseId;
}

inline
unsigned XtcData::PulseId::control() const
{
    return (_value & s_cntrl) >> v_cntrl;
}

inline
XtcData::PulseId& XtcData::PulseId::operator=(const XtcData::PulseId& input)
{
    _value = input._value;
    return *this;
}

inline
bool XtcData::PulseId::operator==(const XtcData::PulseId& ref) const
{
    return value() == ref.value();
}

inline
bool XtcData::PulseId::operator!=(const XtcData::PulseId& ref) const
{
    return value() != ref.value();
}

inline
bool XtcData::PulseId::operator>=(const XtcData::PulseId& ref) const
{
    return value() >= ref.value();
}

inline
bool XtcData::PulseId::operator<=(const XtcData::PulseId& ref) const
{
    return value() <= ref.value();
}

inline
bool XtcData::PulseId::operator>(const XtcData::PulseId& ref) const
{
    return value() > ref.value();
}

inline
bool XtcData::PulseId::operator<(const XtcData::PulseId& ref) const
{
    return value() < ref.value();
}
}

#endif
