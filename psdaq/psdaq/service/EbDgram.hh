#ifndef PSDAQ_EBDGRAM_H
#define PSDAQ_EBDGRAM_H

#include "xtcdata/xtc/Dgram.hh"

namespace Pds {

#pragma pack(push,4)

class TimingHeader;

class PulseId {
public:
    PulseId(uint64_t value) : _pulseIdAndControl(value) {}
    // mask off 56 bits, since upper 8 bits can have
    // "control" information from the timing system
    // give methods "timing_" prefix to avoid conflict with
    // methods in Transition
    uint64_t                     pulseId()        const {return _pulseIdAndControl&0x00ffffffffffffff;}
private:
    friend TimingHeader;
    unsigned                     timing_control() const {return (_pulseIdAndControl>>56)&0xff;}
    XtcData::TransitionId::Value timing_service() const {return (XtcData::TransitionId::Value)(timing_control()&0xf);}
protected:
    mutable uint64_t _pulseIdAndControl; // Mutable so the EOL bit can be set
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
    bool                          error()   const { return control() & (1 << 7); }
public:
    uint32_t evtCounter;
    uint32_t _opaque[2];
};

class EbDgram : public PulseId, public XtcData::Dgram {
public:
    EbDgram(const PulseId& pulseId, const Dgram& dgram) : PulseId(pulseId.pulseId()), Dgram(dgram) {}
    EbDgram(const TimingHeader& th, const XtcData::Src src, uint32_t envRogMask) : PulseId(th.pulseId()) {
        xtc = {{XtcData::TypeId::Parent, 0}, {src}}; // set the src field for the event builders
        time = th.time;
        // move the control bits from the pulseId into the top 8 bits of env.
        env = (th.control()<<24) | (th.env & envRogMask); // filter out other partition ROGs
    }
public:
    void setEOL() const { _pulseIdAndControl |= 1ULL << (6 + 56); }
    bool isEOL()  const { return (_pulseIdAndControl & (1ULL << (6 + 56))) != 0; }
};

#pragma pack(pop)

}

#endif
