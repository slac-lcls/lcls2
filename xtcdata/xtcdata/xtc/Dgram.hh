#ifndef XtcData_Dgram_hh
#define XtcData_Dgram_hh

#include "Sequence.hh"
#include "Xtc.hh"
#include <stdint.h>

#pragma pack(push,4)

namespace XtcData
{

class Transition {
public:
    Transition() {}
    Transition(const Sequence& seq_, uint32_t env_) {
        seq = seq_;
        env = env_;
    }
    Sequence seq;
    uint32_t env;
};

class Dgram : public Transition {
public:
    Dgram() {}
    Dgram(const Transition& transition_, const Xtc& xtc_) :
        Transition(transition_), xtc(xtc_)  { }
public:
    Xtc xtc;
};

class L1Dgram : public Dgram {
public:
    uint16_t trigLines()     const { return (env>>16)&0xffff; }
    uint16_t readoutGroups() const { return (env)&0xffff; }
};

}

#pragma pack(pop)

#endif
