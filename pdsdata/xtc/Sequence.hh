#ifndef PDS_SEQUENCE_HH
#define PDS_SEQUENCE_HH

#include "pdsdata/xtc/ClockTime.hh"
#include "pdsdata/xtc/TimeStamp.hh"
#include "pdsdata/xtc/TransitionId.hh"

namespace Pds {
  class Sequence {
  public:
    enum Type    {Event = 0, Occurrence = 1, Marker = 2};
    enum         {NumberOfTypes = 3};

  public:
    Sequence() {}
    Sequence(const Sequence&);
    Sequence(const ClockTime& clock, const TimeStamp& stamp);
    Sequence(Type, TransitionId::Value, const ClockTime&, const TimeStamp&);

  public:
    Type type() const;
    TransitionId::Value  service() const;
    bool isExtended() const;
    bool isEvent() const;

  public:
    const ClockTime& clock() const {return _clock;}
    const TimeStamp& stamp() const {return _stamp;}

  public:
    Sequence& operator=(const Sequence&);

  private:
    ClockTime _clock;
    TimeStamp _stamp;
  };
}

#endif
