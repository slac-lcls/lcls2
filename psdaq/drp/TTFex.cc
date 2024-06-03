#include "TTFex.hh"
#include "xtcdata/xtc/DescData.hh"


#define COPY_BITS(var,name,len) {                                       \
        var = 0;                                                        \
        uint8_t* a = descdata.get_array<uint8_t> (descdata.nameindex().nameMap()[#name]).data(); \
        for(unsigned i=0; i<len; i++)                                   \
            if (a[i]) var |= (1<<i);                                    \
    }

Drp::EventInfo::EventInfo(XtcData::DescData& descdata) {
    _pulseId     = descdata.get_value<uint64_t>("pulseId");
    COPY_BITS(_fixedRates,fixedRates,10);
    COPY_BITS(_acRates,acRates,6);
    _beamPresent = descdata.get_value<uint8_t>("ebeamPresent") ? 1:0;
    _beamDestn   = descdata.get_value<uint8_t>("ebeamDestn")&0xf;
    memcpy(_seqInfo, descdata.get_array<uint8_t>(descdata.nameindex().nameMap()["sequenceValues"]).data(), 18*sizeof(uint16_t));
}
