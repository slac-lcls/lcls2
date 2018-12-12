#ifndef PDS_SEQUENCE_HH
#define PDS_SEQUENCE_HH

#include "xtcdata/xtc/TimeStamp.hh"
#include "xtcdata/xtc/PulseId.hh"
#include "xtcdata/xtc/TransitionId.hh"

namespace XtcData
{
    // NB: the other control bits are defined in the .cc file
    enum { v_batch = 7, k_batch = 1 };
    enum { m_batch = ((1 << k_batch) - 1), s_batch = (m_batch << v_batch) };

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
    bool isFirst() const;
    void first();

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
