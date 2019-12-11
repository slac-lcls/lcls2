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
    uint64_t                     pulseId()        const {return _pulseIdAndControl&0x00ffffffffffffff;}
protected:
    unsigned                     timing_control() const {return (_pulseIdAndControl>>56)&0xff;}
    XtcData::TransitionId::Value timing_service() const {return (XtcData::TransitionId::Value)(timing_control()&0xf);}
protected:
    uint64_t _pulseIdAndControl;
};

class TimingHeader : public PulseId, public XtcData::TransitionBase {
public:
    // Don't allow constructing or copying of the memory mapped class
    TimingHeader() = delete;
    TimingHeader(const TimingHeader&) = delete;
    void operator=(const TimingHeader&) = delete;
public:
    unsigned                      control() const { return timing_control(); }
    XtcData::TransitionBase::Type type()    const { return XtcData::TransitionBase::Type((control()>>4)&0x3); }
    XtcData::TransitionId::Value  service() const { return timing_service(); }
    bool                          isEvent() const { return service()==XtcData::TransitionId::L1Accept; }
public:
    uint32_t evtCounter;
    uint32_t _opaque[2];
};

class EbDgram : public PulseId, public XtcData::Dgram {
public:
    EbDgram(uint64_t value, const Dgram& dgram) : PulseId(value), Dgram(dgram) {}
    EbDgram(const TimingHeader& th, const XtcData::Src src, uint32_t envRogMask) : PulseId(th) {
        XtcData::TypeId tid(XtcData::TypeId::Parent, 0);
        xtc.src = src; // set the src field for the event builders
        xtc.damage = 0;
        xtc.contains = tid;
        xtc.extent = sizeof(XtcData::Xtc);
        time = th.time;
        // move the control bits from the pulseId into the top 8 bits of env.
        env = (th.control()<<24) | (th.env & envRogMask); // filter out other partition ROGs
    }
};

#pragma pack(pop)

}

#endif
