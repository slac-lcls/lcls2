#include "xtcdata/xtc/TimeStamp.hh"

/* bit field access enums
*       v is the index of the rightmost bit
*       k is the number bits in the field
*       m is the mask, right justified
*       s is the mask shifted into place
*/

namespace XtcData
{
    enum { v_pulseId = 0, k_pulseId = 56 };
    enum { v_cntrl = 56, k_cntrl = 8 };
    enum { m_pulseId = ((1 << k_pulseId) - 1), s_pulseId = (m_pulseId << v_pulseId) };
    enum { m_cntrl = ((1 << k_cntrl) - 1), s_cntrl = (m_cntrl << v_cntrl) };
}

XtcData::TimeStamp::TimeStamp() : _value(0)
{
}

XtcData::TimeStamp::TimeStamp(const XtcData::TimeStamp& input)
: _value(input._value)
{
}

XtcData::TimeStamp::TimeStamp(const XtcData::TimeStamp& input, unsigned control)
: _value((input._value & s_ticks) | ((control & m_cntrl) << v_cntrl))
{
}

unsigned XtcData::TimeStamp::pulseId() const
{
    return (_value & s_pulseId) >> v_pulseId;
}

unsigned XtcData::TimeStamp::control() const
{
    return (_value & s_cntrl) >> v_cntrl;
}

XtcData::TimeStamp& XtcData::TimeStamp::operator=(const XtcData::TimeStamp& input)
{
    _value = input._value;
    return *this;
}

bool XtcData::TimeStamp::operator==(const XtcData::TimeStamp& ref) const
{
    return pulseId() == ref.pulseId();
}

bool XtcData::TimeStamp::operator!=(const XtcData::TimeStamp& ref) const
{
    return pulseId() != ref.pulseId();
}

bool XtcData::TimeStamp::operator>=(const XtcData::TimeStamp& ref) const
{
    return pulseId() >= ref.pulseId();
}

bool XtcData::TimeStamp::operator<=(const XtcData::TimeStamp& ref) const
{
    return pulseId() <= ref.pulseId();
}

bool XtcData::TimeStamp::operator>(const XtcData::TimeStamp& ref) const
{
    return pulseId() > ref.pulseId();
}

bool XtcData::TimeStamp::operator<(const XtcData::TimeStamp& ref) const
{
    return pulseId() < ref.pulseId();
}
