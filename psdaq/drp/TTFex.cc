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

bool Drp::EventSelect::select(const Drp::EventInfo& info) const
{
    bool l_incl_eventcode = info.eventCode  (incl_eventcode);
    bool l_incl_dest      = info.destination(incl_destination);
    bool l_excl_eventcode = info.eventCode  (excl_eventcode);
    bool l_excl_dest      = info.destination(excl_destination);

    return (l_incl_eventcode || l_incl_dest) and not 
        (l_excl_eventcode || l_excl_dest);
}
