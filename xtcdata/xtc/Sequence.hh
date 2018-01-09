#ifndef PDS_SEQUENCE_HH
#define PDS_SEQUENCE_HH

#include "xtcdata/xtc/TimeStamp.hh"
#include "xtcdata/xtc/PulseId.hh"
#include "xtcdata/xtc/TransitionId.hh"

namespace XtcData
{
class Sequence
{
public:
    enum Type { Event = 0, Occurrence = 1, Marker = 2 };
    enum { NumberOfTypes = 3 };

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
    bool isExtended() const;
    bool isEvent() const;

public:
    const TimeStamp& stamp() const
    {
        return _stamp;
    }
    const PulseId& pulseId() const
    {
        return _pulseId;
    }

public:
    Sequence& operator=(const Sequence&);

private:
    PulseId   _pulseId;
    TimeStamp _stamp;
};
}

#endif
