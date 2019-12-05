#ifndef XtcData_Dgram_hh
#define XtcData_Dgram_hh

#include "TimeStamp.hh"
#include "TransitionId.hh"
#include "Xtc.hh"
#include <stdint.h>

#pragma pack(push,4)

namespace XtcData
{

class Transition {
public:
    enum Type { Event = 0, Occurrence = 1, Marker = 2 };
    enum { NumberOfTypes = 3 };
    Transition() {}
    Transition(Type type_, TransitionId::Value tid_,
               const TimeStamp& time_, uint32_t env_) :
        time(time_), env((type_<<28)|(tid_<<24)|(env_&0xffffff)) {}
public:
    uint16_t readoutGroups()      const { return (env)&0xffff; }
    unsigned control()            const { return (env>>24)&0xff; }
    Type type()                   const { return Type((control()>>4)&0x3); }
    TransitionId::Value service() const { return TransitionId::Value(control()&0xf); }
    bool isEvent()                const { return service()==TransitionId::L1Accept; }
public:
    TimeStamp time;
    uint32_t env;
};

class Dgram : public Transition {
public:
    static const unsigned MaxSize = 0x1000000;
    Dgram() {}
    Dgram(const Transition& transition_) :
        Transition(transition_) { }
    Dgram(const Transition& transition_, const Xtc& xtc_) :
        Transition(transition_), xtc(xtc_)  { }
public:
    Xtc xtc;
};

// PulseId and EbDgram should move to psdaq

class PulseId {
public:
    PulseId(uint64_t value) : _value(value) {}
    // mask off 56 bits, since upper 8 bits can have
    // "control" information from the timing system
    // give methods "timing_" prefix to avoid conflict with
    // methods in Transition
    unsigned timing_control() const {return (_value>>56)&0xff;}
    XtcData::TransitionId::Value timing_service() const {return (XtcData::TransitionId::Value)(timing_control()&0xf);}
    uint64_t pulseId() const {return _value&0x00ffffffffffffff;}
public:
    // FIXME: take away "public" and rename (cpo)
    uint64_t _value;
};

// FIXME: move into psdaq (cpo)
class EbDgram : public PulseId, public Dgram {
public:
    EbDgram(uint64_t value, Dgram dgram) : PulseId(value), Dgram(dgram) {}
    EbDgram(uint64_t value) : PulseId(value) {}
};

class L1Dgram : public Dgram {
public:
    // 8 reserved bits.  Perhaps for future trigger lines?
    uint16_t reserved() const { return (env>>16)&0xff; }
};

}

#pragma pack(pop)

#endif
