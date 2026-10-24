/**
 * @file
 * @brief TimingData, a packed timing record, and TimingDef, its XTC name list with helpers that write timing data into an Xtc.
 */
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/NamesLookup.hh"

#include <stdint.h>

namespace XtcData {
    class Xtc;
    class NamesId;
};

/** Namespace of the psdaq data reduction pipeline (drp) code. */
namespace Drp {
#pragma pack(push,1)
    /** Packed (1-byte aligned) 153-byte timing record whose fields follow the order of the TimingDef name list. Field meanings below come from the names only unless stated otherwise. */
    class TimingData {
    public:
        /** Indices into fixedRates for fixedRate(); the rates are taken from the enumerator names. */
        enum FixedRates { _1Hz, /**< Index 0 (1 Hz by name). */ _10Hz, /**< Index 1 (10 Hz by name). */ _100Hz, /**< Index 2 (100 Hz by name). */ _1kHz, /**< Index 3 (1 kHz by name). */ _10kHz, /**< Index 4 (10 kHz by name). */ _71kHz, /**< Index 5 (71 kHz by name). */ _1MHz  /**< Index 6 (1 MHz by name). */ };
        /** Return true if fixedRates[r] is non-zero. */
        bool fixedRate(FixedRates r) const { return fixedRates[r]!=0; }
        /** Return true if bit (code & 0xf) of sequenceValues[code >> 4] is set. */
        bool eventCode(unsigned code) const { return (sequenceValues[code>>4]&(1<<(code&0xf)))!=0; }
    public:
        uint64_t pulseId;  ///< Pulse ID (64 bits).
        uint64_t timeStamp;  ///< Timestamp (64 bits).
        uint8_t  fixedRates[10];  ///< Ten fixed-rate flags, indexed by FixedRates; non-zero means set (see fixedRate()).
        uint8_t  acRates[6];  ///< Six AC-rate flags. Inferred from the name; not verified in code.
        uint8_t  timeSlot;  ///< Time slot. Inferred from the name; not verified in code.
        uint16_t timeSlotPhase;  ///< Time slot phase. Inferred from the name; not verified in code.
        uint8_t  ebeamPresent;  ///< Electron-beam-present flag. Inferred from the name; not verified in code.
        uint8_t  ebeamDestn;  ///< Electron beam destination. Inferred from the name; not verified in code.
        uint16_t ebeamCharge;  ///< Electron beam charge (16-bit raw value). Inferred from the name; not verified in code.
        uint16_t ebeamEnergy[4];  ///< Four electron beam energy values (16-bit raw). Inferred from the name; not verified in code.
        uint16_t xWavelength[2];  ///< Two X-ray wavelength values (16-bit raw). Inferred from the name; not verified in code.
        uint16_t dmod5;  ///< 16-bit value named dmod5; meaning not shown in code.
        uint8_t  mpsLimits[16];  ///< Sixteen MPS limit values. Inferred from the name; not verified in code.
        uint8_t  mpsPowerClass[16];  ///< Sixteen MPS power class values. Inferred from the name; not verified in code.
        uint16_t sequenceValues[18];  ///< Eighteen 16-bit sequence words; eventCode() reads event code bits from them.
        uint32_t inhibitCounts[8];  ///< Eight 32-bit inhibit counters. Inferred from the name; not verified in code.
    };
#pragma pack(pop)

    /** XtcData::VarDef with the names and types of the 16 timing fields, in the order of the index enum, plus static helpers that write timing data into an Xtc. */
    class TimingDef : public XtcData::VarDef
    {
    public:
        /** Size constant for the timing record. */
        enum { data_size = 153  /**< Size in bytes of the packed TimingData layout (153); describeData() copies this many bytes. */ };
        /** Positions of the fields in the TimingDef name list. */
        enum index {
            pulseId,  ///< Index 0: pulseId (UINT64).
            timeStamp,  ///< Index 1: timeStamp (UINT64).
            fixedRates,  ///< Index 2: fixedRates (UINT8 array of 10).
            acRates,  ///< Index 3: acRates (UINT8 array of 6).
            timeSlot,  ///< Index 4: timeSlot (UINT8).
            timeSlotPhase,  ///< Index 5: timeSlotPhase (UINT16).
            ebeamPresent,  ///< Index 6: ebeamPresent (UINT8).
            ebeamDestn,  ///< Index 7: ebeamDestn (UINT8).
            ebeamCharge,  ///< Index 8: ebeamCharge (UINT16).
            ebeamEnergy,  ///< Index 9: ebeamEnergy (UINT16 array of 4).
            xWavelength,  ///< Index 10: xWavelength (UINT16 array of 2).
            dmod5,  ///< Index 11: dmod5 (UINT16).
            mpsLimits,  ///< Index 12: mpsLimits (UINT8 array of 16).
            mpsPowerClass,  ///< Index 13: mpsPowerClass (UINT8 array of 16).
            sequenceValues,  ///< Index 14: sequenceValues (UINT16 array of 18).
            inhibitCounts,  ///< Index 15: inhibitCounts (UINT32 array of 8).
        };
        /** Fill NameVec with the 16 timing fields: UINT64 scalars pulseId and timeStamp, then fixedRates, acRates, timeSlot, timeSlotPhase, ebeamPresent, ebeamDestn, ebeamCharge, ebeamEnergy, xWavelength, dmod5, mpsLimits, mpsPowerClass, sequenceValues and inhibitCounts with the types listed in the index enum. */
        TimingDef();

        /** Append a DescribedData to the Xtc that holds a copy of the first data_size (153) bytes of the last argument, and set the shapes of the eight array fields (10, 6, 4, 2, 16, 16, 18 and 8 elements). */
        static void describeData   (XtcData::Xtc&, const void* bufEnd, XtcData::NamesLookup&, XtcData::NamesId, const uint8_t*);
        /** Append CreateData values decoded from a raw timing record (last argument): pulse ID and timestamp, rate flags expanded from bits of a 16-bit word, a 3-bit time slot and 13-bit phase, beam present and destination bits, charge, the energy and wavelength arrays, dmod5, MPS limit bits, 4-bit MPS power classes, sequence values and inhibit counts. */
        static void createDataNoBSA(XtcData::Xtc&, const void* bufEnd, XtcData::NamesLookup&, XtcData::NamesId, const uint8_t*);
        /** Append CreateData values taken from a header (pulse ID and timestamp; fifth argument) and an ETM record (rate flags, time slot, beam present and destination bits, sequence values; sixth argument). Fields not taken from either (phase, charge, energy, wavelength, dmod5, MPS values, inhibit counts) are written as zeros. */
        static void createDataETM  (XtcData::Xtc&, const void* bufEnd, XtcData::NamesLookup&, XtcData::NamesId, const uint8_t*, const uint8_t*);
    };
};
