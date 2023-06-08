#ifndef BldNames_hh
#define BldNames_hh

#include "xtcdata/xtc/VarDef.hh"

namespace BldNames {
    class EBeamDataV7 : public XtcData::VarDef {
    public:
        EBeamDataV7();
    };
    class PCav : public XtcData::VarDef {
    public:
        PCav();
    };
    class GmdV1 : public XtcData::VarDef {
    public:
        GmdV1();
    };
    class IpimbDataV2 : public XtcData::VarDef {
    public:
        IpimbDataV2();
    };
    class IpimbConfigV2 : public XtcData::VarDef {
    public:
        IpimbConfigV2();
    };
    class GmdV2 : public XtcData::VarDef { 
    public:
        GmdV2();
    };

    class XGmdV2 : public XtcData::VarDef {
    public:
        XGmdV2();
    };
};

#endif
