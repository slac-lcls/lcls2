#ifndef PSDAQ_TIMING_HEADER_H
#define PSDAQ_TIMING_HEADER_H

#include "xtcdata/xtc/Dgram.hh"

namespace Pds {

#pragma pack(push,4)

class TimingHeader : public XtcData::PulseId, public XtcData::Transition {
public:
    // give methods "timing_" prefix to avoid conflict with
    // methods in Transition
    unsigned timing_control() const {return (_value>>56)&0xff;}
    XtcData::TransitionId::Value timing_service() const {return (XtcData::TransitionId::Value)(timing_control()&0xf);}
    uint32_t evtCounter;
    uint32_t _opaque[2];
};

#pragma pack(pop)

}

#endif
