#ifndef PSDAQ_EBDGRAM_H
#define PSDAQ_EBDGRAM_H

#include "xtcdata/xtc/Dgram.hh"

namespace Pds {

#pragma pack(push,4)

class PulseId {
public:
    PulseId(uint64_t value) : _pulseIdAndControl(value) {}
    // mask off 56 bits, since upper 8 bits can have
    // "control" information from the timing system
    // give methods "timing_" prefix to avoid conflict with
    // methods in Transition
    unsigned timing_control() const {return (_pulseIdAndControl>>56)&0xff;}
    XtcData::TransitionId::Value timing_service() const {return (XtcData::TransitionId::Value)(timing_control()&0xf);}
    uint64_t pulseId() const {return _pulseIdAndControl&0x00ffffffffffffff;}
protected:
    uint64_t _pulseIdAndControl;
};

class TimingHeader : public PulseId, public XtcData::Transition {
public:
    uint32_t evtCounter;
    uint32_t _opaque[2];
};

class EbDgram : public PulseId, public XtcData::Dgram {
public:
    EbDgram(uint64_t value, Dgram dgram) : PulseId(value), Dgram(dgram) {}
    EbDgram(TimingHeader& th, XtcData::Src src) : PulseId(th) {
            XtcData::TypeId tid(XtcData::TypeId::Parent, 0);
            xtc.src = src; // set the src field for the event builders
            xtc.damage = 0;
            xtc.contains = tid;
            xtc.extent = sizeof(XtcData::Xtc);
            time = th.time;
            env = (th.env&0xffffff) | (th.service()<<24);
    }
};

#pragma pack(pop)

}

#endif
