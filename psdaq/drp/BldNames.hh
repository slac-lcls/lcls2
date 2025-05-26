#ifndef BldNames_hh
#define BldNames_hh

#include "xtcdata/xtc/VarDef.hh"
#include <string>
#include <vector>
#include <map>

namespace BldNames {
    class UsdUsbDataV1 : public XtcData::VarDef {
    public:
        UsdUsbDataV1();
        static unsigned mcaddr(const char*);
        static std::vector<unsigned> arraySizes();
    };
    class SpectrometerDataV1 : public XtcData::VarDef {
    public:
        SpectrometerDataV1();
        // Entry [X]=Y means entry X has shape determined by value of entry Y
        static std::map<unsigned,unsigned> arraySizeMap();
        static std::vector<unsigned> byteSizes();
    };
    class EBeamDataV7 : public XtcData::VarDef {
    public:
        EBeamDataV7();
    };
    class PCav : public XtcData::VarDef {
    public:
        PCav();
    };
    class GasDet : public XtcData::VarDef {
    public:
        GasDet();
    };
    class BeamMonitorV1 : public XtcData::VarDef {
    public:
        BeamMonitorV1();
        static unsigned mcaddr(const char*);
        static std::vector<unsigned> arraySizes();
    };
    class GmdV1 : public XtcData::VarDef {
    public:
        GmdV1();
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
