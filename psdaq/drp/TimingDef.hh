#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/NamesLookup.hh"

#include <stdint.h>

namespace XtcData {
    class Xtc;
    class NamesId;
};

namespace Drp {
    class TimingDef : public XtcData::VarDef
    {
    public:
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
        };
        TimingDef();

        static void describeData   (XtcData::Xtc&, XtcData::NamesLookup&, XtcData::NamesId, const uint8_t*);
        static void createDataNoBSA(XtcData::Xtc&, XtcData::NamesLookup&, XtcData::NamesId, const uint8_t*);
        static void createDataETM  (XtcData::Xtc&, XtcData::NamesLookup&, XtcData::NamesId, const uint8_t*, const uint8_t*);
    };
};
