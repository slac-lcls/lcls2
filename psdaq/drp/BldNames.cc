#include "BldNames.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/ShapesData.hh"

using namespace XtcData;

BldNames::UsdUsbDataV1::UsdUsbDataV1() {
    // Config
    NameVec.push_back(Name("_countMode",     Name::UINT32, 1)); // Array: [4]
    NameVec.push_back(Name("_quadMode",      Name::UINT32, 1)); // Array: [4]
    // FEX Config
    NameVec.push_back(Name("_offset",        Name::INT32, 1));  // Array: [4]
    NameVec.push_back(Name("_scale",         Name::DOUBLE, 1)); // Array: [4]
    NameVec.push_back(Name("_name",          Name::INT8, 1));   // Array: [4][48] (CHARSTR)
    // Data
    NameVec.push_back(Name("_header",        Name::UINT8, 1));  // Array: [6]
    NameVec.push_back(Name("_din",           Name::UINT8));
    NameVec.push_back(Name("_estop",         Name::UINT8));
    NameVec.push_back(Name("_timestamp",     Name::UINT32));
    NameVec.push_back(Name("_count",         Name::UINT32, 1)); // Array: [4]
    NameVec.push_back(Name("_status",        Name::UINT8, 1));  // Array: [4]
    NameVec.push_back(Name("_ain",           Name::UINT16, 1)); // Array: [4]
    // FEX
    NameVec.push_back(Name("encoder_values", Name::DOUBLE, 1)); // Array: [4]
}

static std::vector<unsigned> _usdUsbArraySizes {
    4, 4,                // Config arrays
    4, 4, 4*48,          // FEX Config arrays
    6, 0, 0, 0, 4, 4, 4, // Data arrays
    4,                   // FEX array
};

std::vector<unsigned> BldNames::UsdUsbDataV1::arraySizes()
{
    return _usdUsbArraySizes;
}

static std::map<std::string, unsigned> _usdUsbMcaddr{
    { "XrtUsbEncoder01", 0xefff1844 },
    { "XppUsbEncoder01", 0xefff1845 },
    { "XppUsbEncoder02", 0xefff1846 },
    { "XcsUsbEncoder01", 0xefff1847 },
    { "CxiUsbEncoder01", 0xefff1848 },
    { "MfxUsbEncoder01", 0xefff185e },
};


unsigned BldNames::UsdUsbDataV1::mcaddr(const char* n)
{
    std::string s(n);
    return _usdUsbMcaddr.find(s) == _usdUsbMcaddr.end() ? 0 : _usdUsbMcaddr[s];
}

BldNames::SpectrometerDataV1::SpectrometerDataV1() {
    /* Width of camera frame (size of hproj) */
    NameVec.push_back(Name("width", Name::UINT32));

    /* First row of pixels used in the projection ROI */
    NameVec.push_back(Name("hproj_y1", Name::UINT32));

    /* Last row of pixels used in the projection ROI */
    NameVec.push_back(Name("hproj_y2", Name::UINT32));

    /* Raw center of mass, no baseline subtraction */
    NameVec.push_back(Name("comRaw", Name::DOUBLE));

    /* Baseline level for calculated values */
    NameVec.push_back(Name("baseline", Name::DOUBLE));

    /* Baseline-subtracted center of mass  */
    NameVec.push_back(Name("com", Name::DOUBLE));

    /* Integrated area under spectrum (no baseline subtraction) */
    NameVec.push_back(Name("integral", Name::DOUBLE));

    /* Number of peak fits performed */
    NameVec.push_back(Name("nPeaks", Name::UINT32));

    /* Projection of spectrum onto energy axis */
    NameVec.push_back(Name("hproj", Name::INT32, 1)); // hproj[width]

    /* Peak position array */
    NameVec.push_back(Name("peakPos", Name::DOUBLE, 1)); // peakPos[nPeaks]

    /* Peak height array */
    NameVec.push_back(Name("peakHeight", Name::DOUBLE, 1)); // peakHeight[nPeaks]

    /* Peak FWHM array */
    NameVec.push_back(Name("FWHM", Name::INT32, 1)); // FWHM[nPeaks]
}

std::vector<unsigned> BldNames::SpectrometerDataV1::byteSizes() {
    std::vector<unsigned> sizes {4, 4, 4, 8, 8, 8, 8, 4, 4, 8, 8, 4};
    return sizes;
}

std::map<unsigned,unsigned> BldNames::SpectrometerDataV1::arraySizeMap() {
    std::map<unsigned,unsigned> sizeMap {
        { 0, 0}, // If map[X]=X this is a scalar
        { 1, 1},
        { 2, 2},
        { 3, 3},
        { 4, 4},
        { 5, 5},
        { 6, 6},
        { 7, 7},
        { 8, 0}, // If map[X]=Y Use map[Y] as array length of map[X]
        { 9, 7},
        {10, 7},
        {11, 7},
    };
    return sizeMap;
}

BldNames::EBeamDataV7::EBeamDataV7() {
    NameVec.push_back(Name("damageMask"       , Name::UINT32));
    NameVec.push_back(Name("ebeamCharge"      , Name::DOUBLE));
    NameVec.push_back(Name("ebeamL3Energy"    , Name::DOUBLE));
    NameVec.push_back(Name("ebeamLTUPosX"     , Name::DOUBLE));
    NameVec.push_back(Name("ebeamLTUPosY"     , Name::DOUBLE));
    NameVec.push_back(Name("ebeamLUTAngX"     , Name::DOUBLE));
    NameVec.push_back(Name("ebeamLTUAngY"     , Name::DOUBLE));
    NameVec.push_back(Name("ebeamPkCurrBC2"   , Name::DOUBLE));
    NameVec.push_back(Name("ebeamEnergyBC2"   , Name::DOUBLE));
    NameVec.push_back(Name("ebeamPkCurrBC1"   , Name::DOUBLE));
    NameVec.push_back(Name("ebeamEnergyBC1"   , Name::DOUBLE));
    NameVec.push_back(Name("ebeamUndPosX"     , Name::DOUBLE));
    NameVec.push_back(Name("ebeamUndPosY"     , Name::DOUBLE));
    NameVec.push_back(Name("ebeamUndAngX"     , Name::DOUBLE));
    NameVec.push_back(Name("ebeamUndAngY"     , Name::DOUBLE));
    NameVec.push_back(Name("ebeamXTCAVAmpl"   , Name::DOUBLE));
    NameVec.push_back(Name("ebeamXTCAVPhase"  , Name::DOUBLE));
    NameVec.push_back(Name("ebeamDumpCharge"  , Name::DOUBLE));
    NameVec.push_back(Name("ebeamPhotonEnergy", Name::DOUBLE));
    NameVec.push_back(Name("ebeamLTU250"      , Name::DOUBLE));
    NameVec.push_back(Name("ebeamLTU450"      , Name::DOUBLE));
}

BldNames::PCav::PCav() {
    NameVec.push_back(Name("fitTime1"      , Name::DOUBLE));
    NameVec.push_back(Name("fitTime2"      , Name::DOUBLE));
    NameVec.push_back(Name("charge1"       , Name::DOUBLE));
    NameVec.push_back(Name("charge2"       , Name::DOUBLE));
}

BldNames::GasDet::GasDet() {
    NameVec.push_back(Name("f11ENRC"      , Name::DOUBLE));
    NameVec.push_back(Name("f12ENRC"      , Name::DOUBLE));
    NameVec.push_back(Name("f21ENRC"      , Name::DOUBLE));
    NameVec.push_back(Name("f22ENRC"      , Name::DOUBLE));
    NameVec.push_back(Name("f63ENRC"      , Name::DOUBLE));
    NameVec.push_back(Name("f64ENRC"      , Name::DOUBLE));
}

BldNames::GmdV1::GmdV1() {
    NameVec.push_back(Name("energy"      , Name::DOUBLE));
    NameVec.push_back(Name("xpos"        , Name::DOUBLE));
    NameVec.push_back(Name("ypos"        , Name::DOUBLE));
    NameVec.push_back(Name("avgIntensity", Name::DOUBLE));
    NameVec.push_back(Name("rmsElectronSum", Name::INT64));
    NameVec.push_back(Name("electron1BkgNoiseAvg", Name::INT16));
    NameVec.push_back(Name("electron2BkgNoiseAvg", Name::INT16));
}

BldNames::BeamMonitorV1::BeamMonitorV1() {
    NameVec.push_back(Name("totalIntensityJoules", Name::DOUBLE));
    NameVec.push_back(Name("xPositionMeters"     , Name::DOUBLE));
    NameVec.push_back(Name("yPositionMeters"     , Name::DOUBLE));
    NameVec.push_back(Name("peakAmplitude"       , Name::DOUBLE,1)); // 16
    NameVec.push_back(Name("peakTime"            , Name::UINT16,1)); // 16
}

static std::vector<unsigned> _bmmonArraySizes { 0, 0, 0, 16, 16 };

std::vector<unsigned> BldNames::BeamMonitorV1::arraySizes() { return _bmmonArraySizes; }

static std::map<std::string,unsigned> _bmmonMcaddr
{ {"MfxBmMon"   ,0xefff183e},
  {"HfxSb1BmMon",0xefff1843},
  {"XcsSndDio"  ,0xefff1849},
  {"MfxUsrDio"  ,0xefff184a},
  {"XppSb2BmMon",0xefff184b},
  {"XppSb3BmMon",0xefff184c},
  {"HfxDg2BmMon",0xefff184d},
  {"XcsSb1BmMon",0xefff184e},
  {"XcsSb2BmMon",0xefff184f},
  {"CxiDg2BmMon",0xefff1850},
  {"CxiDg3BmMon",0xefff1851},
  {"MfxDg1BmMon",0xefff1852},
  {"MfxDg2BmMon",0xefff1853},
  {"MecXt2BmMon02",0xefff1857},
  {"MecXt2BmMon03",0xefff1858},
  {"XppUsrDio"  ,0xefff1859},
  {"XppAlcDio"  ,0xefff185a},
  {"XcsUsrDio"  ,0xefff185b},
  {"CxiUsrDio"  ,0xefff185c},
  {"MecUsrDio"  ,0xefff185d}, };


unsigned BldNames::BeamMonitorV1::mcaddr(const char* n)
{
    std::string s(n);
    return _bmmonMcaddr.find(s) == _bmmonMcaddr.end() ? 0 : _bmmonMcaddr[s];
}


BldNames::GmdV2::GmdV2() {
    NameVec.push_back(Name("millijoulesperpulse",Name::FLOAT,0));
    NameVec.push_back(Name("RMS_E1",Name::FLOAT,0));
}

BldNames::XGmdV2::XGmdV2() {
    NameVec.push_back(Name("millijoulesperpulse",Name::FLOAT,0));
    NameVec.push_back(Name("POSY",Name::FLOAT,0));
    NameVec.push_back(Name("RMS_E1",Name::FLOAT,0));
    NameVec.push_back(Name("RMS_E2",Name::FLOAT,0));
}

