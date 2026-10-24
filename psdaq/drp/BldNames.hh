/**
 * @file
 * @brief XtcData::VarDef name lists for BLD record types (UsdUsb encoder, spectrometer, EBeam, PCav, GasDet, beam monitor, GMD and XGMD).
 */
#ifndef BldNames_hh
#define BldNames_hh

#include "xtcdata/xtc/VarDef.hh"
#include <string>
#include <vector>
#include <map>

/** XtcData::VarDef name lists for BLD record types; each constructor fills NameVec with the field names and types in record order. */
namespace BldNames {
    /** Name list for UsdUsbDataV1 records: configuration, FEX configuration, data and FEX fields (13 entries). */
    class UsdUsbDataV1 : public XtcData::VarDef {
    public:
        /** Fill NameVec with _countMode, _quadMode, _offset, _scale, _name, _header, _din, _estop, _timestamp, _count, _status, _ain and encoder_values. */
        UsdUsbDataV1();
        /** Return the address listed for device name n in a fixed table (0xefff18xx values, for example XppUsbEncoder01 gives 0xefff1845), or 0 if the name is not in the table. */
        static unsigned mcaddr(const char*);
        /** Return the array length of each NameVec entry in order (0 for scalars): 4, 4, 4, 4, 192, 6, 0, 0, 0, 4, 4, 4, 4. */
        static std::vector<unsigned> arraySizes();
    };
    /** Name list for SpectrometerDataV1 records: scalar results followed by the hproj, peakPos, peakHeight and FWHM arrays. */
    class SpectrometerDataV1 : public XtcData::VarDef {
    public:
        /** Fill NameVec with width, hproj_y1, hproj_y2, comRaw, baseline, com, integral, nPeaks, and the arrays hproj (INT32), peakPos and peakHeight (DOUBLE) and FWHM (INT32). */
        SpectrometerDataV1();
        // Entry [X]=Y means entry X has shape determined by value of entry Y
        /** Return a map from entry X to entry Y: X is a scalar if Y equals X, otherwise the value of entry Y is the array length of X (hproj uses width, the three peak arrays use nPeaks). */
        static std::map<unsigned,unsigned> arraySizeMap();
        /** Return the element size in bytes of each entry in order: 4, 4, 4, 8, 8, 8, 8, 4, 4, 8, 8, 4. */
        static std::vector<unsigned> byteSizes();
    };
    /** Name list for EBeamDataV7 records: a UINT32 damageMask followed by 20 DOUBLE fields. */
    class EBeamDataV7 : public XtcData::VarDef {
    public:
        /** Fill NameVec with damageMask (UINT32) and the 20 DOUBLE fields from ebeamCharge to ebeamLTU450. */
        EBeamDataV7();
    };
    /** Name list for PCav records: four DOUBLE fields. */
    class PCav : public XtcData::VarDef {
    public:
        /** Fill NameVec with fitTime1, fitTime2, charge1 and charge2 (all DOUBLE). */
        PCav();
    };
    /** Name list for GasDet records: six DOUBLE fields. */
    class GasDet : public XtcData::VarDef {
    public:
        /** Fill NameVec with f11ENRC, f12ENRC, f21ENRC, f22ENRC, f63ENRC and f64ENRC (all DOUBLE). */
        GasDet();
    };
    /** Name list for BeamMonitorV1 records: three DOUBLE scalars and two 16-element arrays. */
    class BeamMonitorV1 : public XtcData::VarDef {
    public:
        /** Fill NameVec with totalIntensityJoules, xPositionMeters, yPositionMeters (DOUBLE), peakAmplitude (DOUBLE array) and peakTime (UINT16 array). */
        BeamMonitorV1();
        /** Return the address listed for device name n in a fixed table (0xefff18xx values, for example MfxBmMon gives 0xefff183e), or 0 if the name is not in the table. */
        static unsigned mcaddr(const char*);
        /** Return the array length of each entry in order: 0, 0, 0, 16, 16. */
        static std::vector<unsigned> arraySizes();
    };
    /** Name list for GmdV1 records. */
    class GmdV1 : public XtcData::VarDef {
    public:
        /** Fill NameVec with energy, xpos, ypos, avgIntensity (DOUBLE), rmsElectronSum (INT64), electron1BkgNoiseAvg and electron2BkgNoiseAvg (INT16). */
        GmdV1();
    };
    /** Name list for GmdV2 records. */
    class GmdV2 : public XtcData::VarDef { 
    public:
        /** Fill NameVec with millijoulesperpulse and RMS_E1 (FLOAT scalars). */
        GmdV2();
    };
    /** Name list for XGmdV2 records. */
    class XGmdV2 : public XtcData::VarDef {
    public:
        /** Fill NameVec with millijoulesperpulse, POSY, RMS_E1 and RMS_E2 (FLOAT scalars). */
        XGmdV2();
    };
};

#endif
