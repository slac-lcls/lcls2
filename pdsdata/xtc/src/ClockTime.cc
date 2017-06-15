#include "pdsdata/xtc/ClockTime.hh"
#include <math.h>

Pds::ClockTime::ClockTime() {}
Pds::ClockTime::ClockTime(const ClockTime& t) : _low(t._low), _high(t._high) {}
Pds::ClockTime::ClockTime(const timespec& ts) : _low(ts.tv_nsec), _high(ts.tv_sec) {}
Pds::ClockTime::ClockTime(unsigned sec, unsigned nsec) : _low(nsec), _high(sec) {}
Pds::ClockTime::ClockTime(double sec)
{
  double intpart;
  double fracpart = modf(sec, &intpart);
  _high = (unsigned) intpart;
  _low = (unsigned) (1.e9 * fracpart + 0.5);
}

bool Pds::ClockTime::isZero() const
{
  return _low == 0 && _high == 0;
}

double Pds::ClockTime::asDouble() const
{
  return _high + _low / 1.e9;
}

Pds::ClockTime& Pds::ClockTime::ClockTime::operator=(const ClockTime& input)
{
  _low  = input._low;
  _high = input._high;
  return *this;
}

bool Pds::ClockTime::ClockTime::operator>(const ClockTime& t) const
{
  return (_high > t._high) || (_high == t._high && _low > t._low);
}

bool Pds::ClockTime::ClockTime::operator==(const ClockTime& t) const
{
  return (_high == t._high) && (_low == t._low);
}
