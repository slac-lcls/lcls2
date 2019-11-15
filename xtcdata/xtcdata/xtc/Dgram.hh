#ifndef XtcData_Dgram_hh
#define XtcData_Dgram_hh

#include "TimeStamp.hh"
#include "TransitionId.hh"
#include "Xtc.hh"
#include <stdint.h>

#pragma pack(push,4)

namespace XtcData
{

class PulseId {
public:
    PulseId(unsigned value) : _value(value) {}
    // mask off 56 bits, since upper 8 bits can have
    // "control" information from the timing system
    uint64_t pulseId() {return _value&0xfffffffffffffff;}
private:
    uint64_t _value;
};

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

class EbDgram;

class Dgram : public Transition {
public:
    static const unsigned MaxSize = 0x1000000;
    Dgram() {}
    Dgram(const Transition& transition_) :
        Transition(transition_) { }
    Dgram(const Transition& transition_, const Xtc& xtc_) :
        Transition(transition_), xtc(xtc_)  { }
    // should be used only in the DAQ to give the online EB access
    // to the pulseId.  This "backs up the pointer" to include pulseid.
    EbDgram& _ebDgram() const {return *(EbDgram*)((char*)this-sizeof(PulseId));}
public:
    Xtc xtc;
};

class EbDgram : public PulseId, public Dgram {
    EbDgram(unsigned value, Dgram dgram) : PulseId(value), Dgram(dgram) {}
};

class L1Dgram : public Dgram {
public:
    // 8 reserved bits.  Perhaps for future trigger lines?
    uint16_t reserved() const { return (env>>16)&0xff; }
};

}

#pragma pack(pop)

#endif
