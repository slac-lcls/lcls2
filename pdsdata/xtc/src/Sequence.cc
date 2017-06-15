#include "pdsdata/xtc/Sequence.hh"

/* bit field access enums
*	v is the index of the rightmost bit
*	k is the number bits in the field
*	m is the mask, right justified
*	s is the mask shifted into place
*/

namespace Pds {
  enum {v_cntrl   = 0, k_cntrl   = 8};
  enum {v_service = 0, k_service = 4};
  enum {v_seqtype = 4, k_seqtype = 2};
  enum {v_extend  = 7, k_extend  = 1};
  
  enum {m_cntrl   = ((1 << k_cntrl)  -1), s_cntrl   = (m_cntrl   << v_cntrl)};
  enum {m_service = ((1 << k_service)-1), s_service = (m_service << v_service)};
  enum {m_seqtype = ((1 << k_seqtype)-1), s_seqtype = (m_seqtype << v_seqtype)};
  enum {m_extend  = ((1 << k_extend )-1), s_extend  = (m_extend  << v_extend)};
}

Pds::Sequence::Sequence(const Pds::Sequence& input) :
  _clock(input._clock),
  _stamp(input._stamp)
{}

Pds::Sequence::Sequence(const ClockTime& clock, const TimeStamp& stamp) :
  _clock(clock),
  _stamp(stamp)
{}

Pds::Sequence::Sequence(Type type,
                        TransitionId::Value service,
                        const ClockTime& clock,
                        const TimeStamp& stamp) :
  _clock(clock),
  _stamp(stamp, 
         ((type & m_seqtype) << v_seqtype) | 
         ((service & m_service) << v_service))
{}

Pds::Sequence::Type Pds::Sequence::type() const
{
  return Type((_stamp.control() >> v_seqtype) & m_seqtype);
}

Pds::TransitionId::Value Pds::Sequence::service() const
{
  return TransitionId::Value((_stamp.control() >> v_service) & m_service);
}

bool Pds::Sequence::isExtended() const
{
  return _stamp.control() & s_extend;
}

bool Pds::Sequence::isEvent() const
{
  return ((_stamp.control() & s_service) >> v_service) == 
    TransitionId::L1Accept;
}

Pds::Sequence& Pds::Sequence::operator=(const Pds::Sequence& input)
{
  _clock = input._clock;
  _stamp = input._stamp;
  return *this;
}
