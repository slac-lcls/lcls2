#pragma once

#include <cstdint>
#include <vector>

namespace XtcData { class DescData; }

namespace Drp {

    class Roi {
    public:
        unsigned x0, y0, x1, y1;
    };

    //
    //  The (l2si-core) EventTimingMessage auxiliary data
    //
    class EventInfo {
    public:
        EventInfo() {}
        EventInfo(XtcData::DescData&);
    public:
        bool fixedRate(unsigned rate) const { return _fixedRates & (1<<rate); }
        bool acRate(unsigned rate,
                    unsigned tslots) const { return _acRates & (1<<rate) && (_timeSlots&tslots); }
        bool eventCode(unsigned ec) const { return _seqInfo[ec>>4] & (1<<(ec&0xf)); }
        bool sequencer(unsigned word,
                       unsigned bit) const { return _seqInfo[word] & (1<<bit); }
    public:
        uint64_t _pulseId; 
        unsigned _fixedRates  : 10;
        unsigned _acRates     : 6;
        unsigned _timeSlots   : 8;
        unsigned _beamPresent : 1, : 3;
        unsigned _beamDestn   : 4;
        uint16_t _seqInfo[18];
    };
};

//
//  Some helpful macros when iterating over a configuration
//
#define COPY_CFG_VECTOR(t,a,b)                                  \
    if (strcmp(name.name(),#a)==0) {                            \
        Array<t> w = descdata.get_array<t>(i);                  \
        std::vector<t>& v = b;                                  \
        unsigned len = w.num_elem();                            \
        v.resize(len);                                          \
        memcpy(v.data(),w.data(),len*sizeof(t));                \
    }
        
