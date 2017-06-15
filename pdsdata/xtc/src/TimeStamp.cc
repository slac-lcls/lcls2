#include "pdsdata/xtc/TimeStamp.hh"

/* bit field access enums
*       v is the index of the rightmost bit
*       k is the number bits in the field
*       m is the mask, right justified
*       s is the mask shifted into place
*/

namespace Pds {
  enum {v_ticks =  0, k_ticks = 24};
  enum {v_cntrl = 24, k_cntrl = 8};
  enum {m_ticks = ((1 << k_ticks)-1), s_ticks = (m_ticks << v_ticks)};
  enum {m_cntrl = ((1 << k_cntrl)-1), s_cntrl = (m_cntrl << v_cntrl)};
  enum {v_fiduc =  0, k_fiduc = TimeStamp::NumFiducialBits};
  enum {v_vecto = TimeStamp::NumFiducialBits, k_vecto = 32-TimeStamp::NumFiducialBits};
  enum {m_fiduc = ((1 << k_fiduc)-1), s_fiduc = (m_fiduc <<v_fiduc)};
  enum {m_vecto = ((1 << k_vecto)-1), s_vecto = (m_vecto <<v_vecto)};
}
 
Pds::TimeStamp::TimeStamp() :
  _low (0),
  _high(0)
{}

Pds::TimeStamp::TimeStamp(const Pds::TimeStamp& input) :
  _low (input._low ),
  _high(input._high)
{}

Pds::TimeStamp::TimeStamp(const Pds::TimeStamp& input, unsigned control) :
  _low((input._low & s_ticks) | ((control & m_cntrl) << v_cntrl)),
  _high(input._high)
{}

Pds::TimeStamp::TimeStamp(unsigned low, unsigned high, unsigned vector, unsigned control) :
  _low ((low  & s_ticks) | ((control & m_cntrl) << v_cntrl)),
  _high((high & s_fiduc) | ((vector  & m_vecto) << v_vecto))
{}

unsigned Pds::TimeStamp::ticks() const
{
  return (_low  & s_ticks) >> v_ticks;
}

unsigned Pds::TimeStamp::fiducials() const
{
  return (_high & s_fiduc) >> v_fiduc;
}

unsigned Pds::TimeStamp::control() const
{
  return (_low & s_cntrl) >> v_cntrl;
}

unsigned Pds::TimeStamp::vector() const
{
  return (_high & s_vecto) >> v_vecto;
}

Pds::TimeStamp& Pds::TimeStamp::operator=(const Pds::TimeStamp& input)
{
  _low  = input._low ;
  _high = input._high;
  return *this;
}

bool Pds::TimeStamp::operator==(const Pds::TimeStamp& ref) const
{
  return fiducials() == ref.fiducials();
}

bool Pds::TimeStamp::operator>=(const Pds::TimeStamp& ref) const
{
  return fiducials() >= ref.fiducials();
}

bool Pds::TimeStamp::operator<=(const Pds::TimeStamp& ref) const
{
  return fiducials() <= ref.fiducials();
}

bool Pds::TimeStamp::operator>(const Pds::TimeStamp& ref) const
{
  return fiducials() > ref.fiducials();
}

bool Pds::TimeStamp::operator<(const Pds::TimeStamp& ref) const
{
  return fiducials() < ref.fiducials();
}
