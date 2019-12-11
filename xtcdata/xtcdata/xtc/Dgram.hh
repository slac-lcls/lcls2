#ifndef XtcData_Dgram_hh
#define XtcData_Dgram_hh

#include "TimeStamp.hh"
#include "TransitionId.hh"
#include "Xtc.hh"
#include <stdint.h>

#pragma pack(push,4)

namespace XtcData
{

class TransitionBase {
public:
    enum Type { Event = 0, Occurrence = 1, Marker = 2 };
    enum { NumberOfTypes = 3 };
    TransitionBase() {}
    TransitionBase(Type type_, TransitionId::Value tid_,
                   const TimeStamp& time_, uint32_t env_) :
        time(time_), env((type_<<28)|(tid_<<24)|(env_&0xffffff)) {}
public:
    uint16_t readoutGroups() const { return (env)&0xffff; }
public:
    TimeStamp time;
    uint32_t env;
};

class Transition : public TransitionBase {
public:
    Transition() {}
    Transition(Type type_, TransitionId::Value tid_,
               const TimeStamp& time_, uint32_t env_) :
      TransitionBase(type_, tid_, time_, env_) {}
public:
    unsigned control()            const { return (env>>24)&0xff; }
    TransitionId::Value service() const { return TransitionId::Value(control()&0xf); }
    Type type()                   const { return Type((control()>>4)&0x3); }
    bool isEvent()                const { return service()==TransitionId::L1Accept; }
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

class L1Dgram : public Dgram {
public:
    // 8 reserved bits.  Perhaps for future trigger lines?
    uint16_t reserved() const { return (env>>16)&0xff; }
};

}

#pragma pack(pop)

#endif
