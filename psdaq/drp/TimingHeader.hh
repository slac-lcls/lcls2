#ifndef PSDAQ_TIMING_HEADER_H
#define PSDAQ_TIMING_HEADER_H

#include "xtcdata/xtc/Sequence.hh"

namespace Pds {

class TimingHeader {
public:
    XtcData::Sequence seq;
    uint32_t evtCounter;
};

}

#endif
