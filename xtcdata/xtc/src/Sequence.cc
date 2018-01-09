#include "xtcdata/xtc/Sequence.hh"

/* bit field access enums
*	v is the index of the rightmost bit
*	k is the number bits in the field
*	m is the mask, right justified
*	s is the mask shifted into place
*/

namespace XtcData
{
enum { v_cntrl = 0, k_cntrl = 8 };
enum { v_service = 0, k_service = 4 };
enum { v_seqtype = 4, k_seqtype = 2 };
enum { v_extend = 7, k_extend = 1 };

enum { m_cntrl = ((1 << k_cntrl) - 1), s_cntrl = (m_cntrl << v_cntrl) };
enum { m_service = ((1 << k_service) - 1), s_service = (m_service << v_service) };
enum { m_seqtype = ((1 << k_seqtype) - 1), s_seqtype = (m_seqtype << v_seqtype) };
enum { m_extend = ((1 << k_extend) - 1), s_extend = (m_extend << v_extend) };
}

XtcData::Sequence::Sequence(const XtcData::Sequence& input)
    : _pulseId(input._pulseId), _stamp(input._stamp)
{
}

XtcData::Sequence::Sequence(const TimeStamp& stamp, const PulseId& pulseId)
    : _pulseId(pulseId), _stamp(stamp)
{
}

XtcData::Sequence::Sequence(Type type, TransitionId::Value service, const TimeStamp& stamp, const PulseId& pulseId)
    : _pulseId(pulseId, ((type & m_seqtype) << v_seqtype) | ((service & m_service) << v_service)), _stamp(stamp)
{
}

XtcData::Sequence::Type XtcData::Sequence::type() const
{
    return Type((_pulseId.control() >> v_seqtype) & m_seqtype);
}

XtcData::TransitionId::Value XtcData::Sequence::service() const
{
    return TransitionId::Value((_pulseId.control() >> v_service) & m_service);
}

bool XtcData::Sequence::isExtended() const
{
    return _pulseId.control() & s_extend;
}

bool XtcData::Sequence::isEvent() const
{
    return ((_pulseId.control() & s_service) >> v_service) == TransitionId::L1Accept;
}

XtcData::Sequence& XtcData::Sequence::operator=(const XtcData::Sequence& input)
{
    _stamp = input._stamp;
    _pulseId = input._pulseId;
    return *this;
}
