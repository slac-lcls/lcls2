#ifndef PDS_SEQUENCE_HH
#define PDS_SEQUENCE_HH

#include "xtcdata/xtc/TimeStamp.hh"
#include "xtcdata/xtc/PulseId.hh"
#include "xtcdata/xtc/TransitionId.hh"

/* bit field access enums
*	v is the index of the rightmost bit
*	k is the number bits in the field
*	m is the mask, right justified
*	s is the mask shifted into place
*/

namespace XtcData
{
    enum { v_cntrl   = 0, k_cntrl   = 8 };
    enum { v_service = 0, k_service = 4 };
    enum { v_seqtype = 4, k_seqtype = 2 };
    enum { v_batch   = 7, k_batch   = 1 };

    enum { m_cntrl   = ((1 << k_cntrl  ) - 1), s_cntrl   = (m_cntrl   << v_cntrl  ) };
    enum { m_service = ((1 << k_service) - 1), s_service = (m_service << v_service) };
    enum { m_seqtype = ((1 << k_seqtype) - 1), s_seqtype = (m_seqtype << v_seqtype) };
    enum { m_batch   = ((1 << k_batch  ) - 1), s_batch   = (m_batch   << v_batch  ) };

class Sequence
{
public:
    enum Type { Event = 0, Occurrence = 1, Marker = 2 };
    enum { NumberOfTypes = 3 };
    enum Batch { IsBatch = s_batch };

public:
    Sequence()
    {
    }
    Sequence(const Sequence&);
    Sequence(const TimeStamp& stamp, const PulseId& pulseId);
    Sequence(Type, TransitionId::Value, const TimeStamp&, const PulseId&);

public:
    Type type() const;
    TransitionId::Value service() const;
    bool isBatch() const;
    void markBatch();
    bool isEvent() const;

public:
    const TimeStamp& stamp() const;
    const PulseId& pulseId() const;

public:
    Sequence& operator=(const Sequence&);

private:
    PulseId   _pulseId;
    TimeStamp _stamp;
};


inline
XtcData::Sequence::Sequence(const XtcData::Sequence& input)
    : _pulseId(input._pulseId), _stamp(input._stamp)
{
}

inline
XtcData::Sequence::Sequence(const TimeStamp& stamp, const PulseId& pulseId)
    : _pulseId(pulseId), _stamp(stamp)
{
}

inline
XtcData::Sequence::Sequence(Type type, TransitionId::Value service, const TimeStamp& stamp, const PulseId& pulseId)
    : _pulseId(pulseId, ((type & m_seqtype) << v_seqtype) | ((service & m_service) << v_service)), _stamp(stamp)
{
}

inline
XtcData::Sequence::Type XtcData::Sequence::type() const
{
    return Type((_pulseId.control() >> v_seqtype) & m_seqtype);
}

inline
XtcData::TransitionId::Value XtcData::Sequence::service() const
{
    return TransitionId::Value((_pulseId.control() >> v_service) & m_service);
}

inline
bool XtcData::Sequence::isBatch() const
{
    return _pulseId.control() & s_batch;
}

inline
void XtcData::Sequence::markBatch()
{
    _pulseId = PulseId(_pulseId, _pulseId.control() | s_batch);
}

inline
bool XtcData::Sequence::isEvent() const
{
    return ((_pulseId.control() & s_service) >> v_service) == TransitionId::L1Accept;
}

inline
const XtcData::TimeStamp& XtcData::Sequence::stamp() const
{
    return _stamp;
}

inline
const XtcData::PulseId& XtcData::Sequence::pulseId() const
{
    return _pulseId;
}

inline
XtcData::Sequence& XtcData::Sequence::operator=(const XtcData::Sequence& input)
{
    _stamp = input._stamp;
    _pulseId = input._pulseId;
    return *this;
}
}

#endif
