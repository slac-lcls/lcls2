#include "Wave8HE.hh"

#include "xtcdata/xtc/Array.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/aes-stream-drivers/DataDriver.h"
#include "psalg/detector/UtilsConfig.hh"
#include "psalg/utils/SysLog.hh"

#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <string>
#include <cmath>

using namespace XtcData;
using logging = psalg::SysLog;
using json = nlohmann::json;

namespace Drp {

  namespace W8HE {
    
    // Helper function: Unpack 24-bit value from densely packed array with sign-extension
    static int32_t unpack24bit(const uint8_t* data, unsigned offset) {
        unsigned byteIdx = offset * 3;  // 3 bytes per value
        uint32_t raw = (data[byteIdx + 2] << 16) | 
                       (data[byteIdx + 1] << 8) | 
                       data[byteIdx];
        // Sign extend from 24-bit to 32-bit
        if (raw & 0x800000) {
            return (int32_t)(raw | 0xFF000000);
        }
        return (int32_t)raw;
    }
    
    class RawStream {
    public:
        static void varDef(VarDef& v, unsigned ch) {
            char name[32];
            // Firmware sends 255 samples (510 bytes) padded to 512 bytes
            // Dynamically size based on actual segment length for flexibility
            sprintf(name, "raw_%d", ch);
            v.NameVec.push_back(XtcData::Name(name, XtcData::Name::UINT16, 1));
        }
        
        static void createData(CreateData& cd, unsigned& index, unsigned ch, Array<uint8_t>& seg) {
            unsigned shape[MaxRank];
            shape[0] = seg.num_elem()>>1;  // Convert bytes to uint16 count
            Array<uint16_t> arrayT = cd.allocate<uint16_t>(index++, shape);
            memcpy(arrayT.data(), seg.data(), seg.num_elem());
        }
    };
    
    class PeakStream {
    public:
        static void varDef(VarDef& v) {  // DIFFERENCE: XtcData:VarDef&
            char name[32];
            
            // Peak positions (8-bit time sample index 0-254)
            for(unsigned i=0; i<8; i++) {
                sprintf(name, "peakPos_%d", i);
                v.NameVec.push_back(XtcData::Name(name, XtcData::Name::UINT8));
            }
            
            // Peak values (24-bit, firmware-scaled by 4)
            // NOTE: Firmware scales by 4 to preserve 0.25-bit precision during
            // integer calculations. We divide by 4.0 to get physical values.
            for(unsigned i=0; i<8; i++) {
                sprintf(name, "peakVal_%d", i);
                v.NameVec.push_back(XtcData::Name(name, XtcData::Name::FLOAT));
            }
            
            // Baselines (24-bit, firmware-scaled by 4)
            for(unsigned i=0; i<8; i++) {
                sprintf(name, "base_%d", i);
                v.NameVec.push_back(XtcData::Name(name, XtcData::Name::FLOAT));
            }
            
            // Peak-based positions
            // NOTE: May be NaN if denominator (xp+xm) is zero. Users should check
            // with std::isnan(). See WAVE8_HE_FIRMWARE.md for details.
            v.NameVec.push_back(XtcData::Name("posPeakX", XtcData::Name::FLOAT));
            v.NameVec.push_back(XtcData::Name("posPeakY", XtcData::Name::FLOAT));
        }
        
        static void createData(CreateData& cd, unsigned& index, Array<uint8_t>& seg) {
            const uint8_t* data = seg.data();
            
            // Extract 8 peak positions from 32-bit words
            // Format: [peakPos:8][peakVal:24]
            for(unsigned i=0; i<8; i++) {
                uint32_t word = *(reinterpret_cast<const uint32_t*>(data + i*4));
                uint8_t peakPos = (word >> 24) & 0xFF;
                cd.set_value(index++, peakPos);
            }
            
            // Extract 8 peak values from 32-bit words and scale to physical values
            // NOTE: Firmware scales values by 4 for precision. Divide by 4.0.
            for(unsigned i=0; i<8; i++) {
                uint32_t word = *(reinterpret_cast<const uint32_t*>(data + i*4));
                int32_t peakVal = word & 0x00FFFFFF;
                
                // Sign extend peak value from 24-bit to 32-bit
                if (peakVal & 0x800000) {
                    peakVal |= 0xFF000000;
                }
                
                cd.set_value(index++, peakVal / 4.0f);
            }
            
            // Unpack 8 densely-packed 24-bit baselines (24 bytes total at offset 32)
            // NOTE: Firmware scales baselines by 4 for precision. Divide by 4.0.
            for(unsigned i=0; i<8; i++) {
                int32_t baseline = unpack24bit(data + 32, i);
                cd.set_value(index++, baseline / 4.0f);
            }
            
            // Extract peak-based X,Y positions (32-bit floats at offset 56)
            float posX = *(reinterpret_cast<const float*>(data + 56));
            float posY = *(reinterpret_cast<const float*>(data + 60));
            
            // Option A: Pass through NaN (default - allows std::isnan() checks)
            cd.set_value(index++, posX);
            cd.set_value(index++, posY);
            
            // Other option: Replace NaN with sentinel value (commented for team discussion)
            // #define WAVE8HE_USE_POSITION_SENTINEL
            // #ifdef WAVE8HE_USE_POSITION_SENTINEL
            //     constexpr float INVALID_POSITION = -999.0f;
            //     float posXSafe = std::isnan(posX) ? INVALID_POSITION : posX;
            //     float posYSafe = std::isnan(posY) ? INVALID_POSITION : posY;
            //     cd.set_value(index++, posXSafe);
            //     cd.set_value(index++, posYSafe);
            // #endif
        }
    };
    
    class IntegralStream {
    public:
        static void varDef(VarDef& v) {
            char name[32];
            
            // Baseline-subtracted integrals (24-bit, firmware-scaled by 4)
            // NOTE: These are BASELINE-SUBTRACTED values and can be NEGATIVE
            // NOTE: Firmware scales integrals by 4 for precision. We divide by 4.0.
            for(unsigned i=0; i<8; i++) {
                sprintf(name, "integral_%d", i);
                v.NameVec.push_back(XtcData::Name(name, XtcData::Name::FLOAT));
            }
            
            // Integral-based positions
            v.NameVec.push_back(XtcData::Name("posIntegralX", XtcData::Name::FLOAT));
            v.NameVec.push_back(XtcData::Name("posIntegralY", XtcData::Name::FLOAT));
        }
        
        static void createData(CreateData& cd, unsigned& index, Array<uint8_t>& seg) {
            const uint8_t* data = seg.data();
            
            // Unpack 8 densely-packed 24-bit integrals (24 bytes total at offset 0)
            // NOTE: These are BASELINE-SUBTRACTED and can be NEGATIVE
            // NOTE: Firmware scales by 4 for precision. Divide by 4.0.
            for(unsigned i=0; i<8; i++) {
                int32_t integral = unpack24bit(data, i);
                cd.set_value(index++, integral / 4.0f);
            }
            
            // Extract integral-based X,Y positions (32-bit floats at offset 24)
            // NOTE: May be NaN if denominator (xp+xm) is zero. Users should check
            // with std::isnan().
            float posX = *(reinterpret_cast<const float*>(data + 24));
            float posY = *(reinterpret_cast<const float*>(data + 28));
            
            // Option A: Pass through NaN (default)
            cd.set_value(index++, posX);
            cd.set_value(index++, posY);
            
            // Other option: Replace NaN with sentinel (see PeakStream for details)
            // #ifdef WAVE8HE_USE_POSITION_SENTINEL
            //     constexpr float INVALID_POSITION = -999.0f;
            //     cd.set_value(index++, std::isnan(posX) ? INVALID_POSITION : posX);
            //     cd.set_value(index++, std::isnan(posY) ? INVALID_POSITION : posY);
            // #endif
        }
    };
    
    class Streams {
    public:
        static void defineData(Xtc& xtc, const void* bufEnd, const char* detName,
                               const char* detType, const char* detNum,
                               NamesLookup& lookup, NamesId& raw, NamesId& fex) {
            // Set up the names for L1Accept data - Raw streams
            {
                Alg alg("raw", 0, 0, 1);
                Names& eventNames = *new(xtc, bufEnd) Names(bufEnd,
                                                            detName, alg,
                                                            detType, detNum, raw);
                VarDef v;
                for(unsigned i=0; i<8; i++)
                    RawStream::varDef(v, i);
                eventNames.add(xtc, bufEnd, v);
                lookup[raw] = NameIndex(eventNames);
            }
            
            // Set up the names for L1Accept data - FEX streams (peak + integral)
            {
                Alg alg("fex", 0, 0, 1);
                Names& eventNames = *new(xtc, bufEnd) Names(bufEnd,
                                                            detName, alg,
                                                            detType, detNum, fex);
                VarDef v;
                PeakStream::varDef(v);
                IntegralStream::varDef(v);
                eventNames.add(xtc, bufEnd, v);
                lookup[fex] = NameIndex(eventNames);
            }
        }
        
        static void createData(XtcData::Xtc&         xtc,
                               const void*           bufEnd,
                               XtcData::NamesLookup& lookup,
                               XtcData::NamesId&     rawId,
                               XtcData::NamesId&     fexId,
                               XtcData::Array<uint8_t>* streams) {
            // Process raw waveform streams
            CreateData raw(xtc, bufEnd, lookup, rawId);
            unsigned index = 0;
            for(unsigned i=0; i<8; i++)
                RawStream::createData(raw, index, i, streams[i]);
            
            // Process FEX streams (peak + integral)
            // NOTE: Skip over empty TDEST slots 0x0A and 0x0B to reach HLS streams at 0x0C and 0x0D
            //   streams[10] = subframes[2+10] = subframes[12] = subframes[0x0C] (HLS Stream 0)
            //   streams[11] = subframes[2+11] = subframes[13] = subframes[0x0D] (HLS Stream 1)
            CreateData fex(xtc, bufEnd, lookup, fexId);
            index = 0;
            
            if (streams[10].data())
                PeakStream::createData(fex, index, streams[10]);
            
            if (streams[11].data())
                IntegralStream::createData(fex, index, streams[11]);
        }
    };

  };  // namespace W8HE

Wave8HE::Wave8HE(Parameters* para, MemPool* pool) :
    BEBDetector(para, pool)
{
    _init(para->kwargs["epics_prefix"].c_str());
}

unsigned Wave8HE::_configure(Xtc& xtc, const void* bufEnd, ConfigIter& configo)
{
    // Set up the names for the event data
    m_evtNamesRaw = NamesId(nodeId, EventNamesIndex+0);
    m_evtNamesFex = NamesId(nodeId, EventNamesIndex+1);

    W8HE::Streams::defineData(xtc, bufEnd,
                              m_para->detName.c_str(),
                              m_para->detType.c_str(),
                              m_para->serNo.c_str(),
                              m_namesLookup,
                              m_evtNamesRaw,
                              m_evtNamesFex);
    return 0;
}

void Wave8HE::_event(XtcData::Xtc& xtc,
                     const void* bufEnd,
                     uint64_t l1count,
                     std::vector< XtcData::Array<uint8_t> >& subframes)
{
    // IMPORTANT: subframes vector is SPARSE, indexed by TDEST value (see BEBDetector.cc:309-312)
    // Vector is pre-sized to max_tdest+1; empty TDEST slots exist but contain null data.
    //
    // Wave8HE TDEST layout:
    //   subframes[0x00 or 0x01] = Timing/Trigger (ignored by detector code)
    //   subframes[0x02-0x09]    = RawWaveform[0-7] (8 channels, 512 bytes each)
    //   subframes[0x0A, 0x0B]   = Empty slots (TDESTs not sent by firmware, but slots exist in vector)
    //   subframes[0x0C]         = Stream 0 (Peak data - 64 bytes, decimal index 12)
    //   subframes[0x0D]         = Stream 1 (Integral data - 32 bytes, decimal index 13)
    //
    // By passing &subframes[2], the streams pointer maps:
    //   streams[0-7]  → subframes[0x02-0x09] (raw waveforms) ✓
    //   streams[8-9]  → subframes[0x0A-0x0B] (empty slots - skipped in Streams::createData)
    //   streams[10]   → subframes[0x0C]      (HLS Stream 0) ✓
    //   streams[11]   → subframes[0x0D]      (HLS Stream 1) ✓
    W8HE::Streams::createData(xtc, bufEnd, m_namesLookup,
                              m_evtNamesRaw, m_evtNamesFex,
                              &subframes[2]);
}

}  // namespace Drp
