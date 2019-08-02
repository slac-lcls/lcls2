#ifndef PSDAQ_TIMING_HEADER_H
#define PSDAQ_TIMING_HEADER_H

#include "xtcdata/xtc/Dgram.hh"

namespace Pds {

#pragma pack(push,4)

  class TimingHeader : public XtcData::Transition {
public:
    uint32_t evtCounter;
    uint32_t _opaque[2];
};

#pragma pack(pop)

}

#endif
