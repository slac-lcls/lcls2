#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/NamesLookup.hh"

#include <stdint.h>

namespace XtcData {
    class Xtc;
    class NamesId;
};

namespace Drp {
    class TimingData {
    public:
        enum FixedRates { _1Hz, _10Hz, _100Hz, _1kHz, _10kHz, _71kHz, _1MHz };
        bool fixedRate(FixedRates r) const { return fixedRates[r]!=0; }
        bool eventCode(unsigned code) const { return (sequenceValues[code>>4]&(1<<(code&0xf)))!=0; }
    public:
        uint64_t pulseId;
        uint64_t timeStamp;
        uint8_t  fixedRates[10];
        uint8_t  acRates[6];
        uint8_t  timeSlot;
        uint16_t timeSlotPhase;
        uint8_t  ebeamPresent;
        uint8_t  ebeamDestn;
        uint16_t ebeamCharge;
        uint16_t ebeamEnergy[4];
        uint16_t xWavelength[2];
        uint16_t dmod5;
        uint8_t  mpsLimits[16];
        uint8_t  mpsPowerClass[16];
        uint16_t sequenceValues[18];
        uint32_t inhibitCounts[8];
    };

    class TimingDef : public XtcData::VarDef
    {
    public:
        enum { data_size = 153 };
        enum index {
            pulseId,
            timeStamp,
            fixedRates,
            acRates,
            timeSlot,
            timeSlotPhase,
            ebeamPresent,
            ebeamDestn,
            ebeamCharge,
            ebeamEnergy,
            xWavelength,
            dmod5,
            mpsLimits,
            mpsPowerClass,
            sequenceValues,
            inhibitCounts,
        };
        TimingDef();

        static void describeData   (XtcData::Xtc&, const void* bufEnd, XtcData::NamesLookup&, XtcData::NamesId, const uint8_t*);
        static void createDataNoBSA(XtcData::Xtc&, const void* bufEnd, XtcData::NamesLookup&, XtcData::NamesId, const uint8_t*);
        static void createDataETM  (XtcData::Xtc&, const void* bufEnd, XtcData::NamesLookup&, XtcData::NamesId, const uint8_t*, const uint8_t*);
    };
};
