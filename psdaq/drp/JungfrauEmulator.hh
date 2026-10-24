/**
 * @file
 * @brief JungfrauEmulator, an XpmDetector that emulates Jungfrau panels (one per PGP lane).
 */
#pragma once

#include "drp.hh"
#include "XpmDetector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesLookup.hh"

#include <vector>
#include <cstdint>
#include <string>

namespace Drp {

/** XpmDetector that emulates Jungfrau panels, one per active PGP lane (detType jungfrauemu in PGPDetectorApp), optionally replaying image data from a file. */
class JungfrauEmulator : public XpmDetector
{
public:
    /** Build synthetic serial numbers for each active lane, read the segNums kwarg (aborting if it does not match the panel count, or if it is missing with more than one panel), and load substitute image data from the file named by the imgArray kwarg if set (aborting if it cannot be opened). */
    JungfrauEmulator(Parameters* para, MemPool* pool);
    /** Run XpmDetector::configure (returning 1 if it fails), then add raw names (one UINT16 rank-3 array named raw) for each panel, with the panel's serial number and its segment number from segNums or, if segNums was not given, detSegment*(PGP_MAX_LANES-1) plus the panel index. Returns 0. */
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) override;
    /** Log an info message and return 0; does nothing else. */
    unsigned beginrun(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) override;
    /** For each active lane among the first PGP_MAX_LANES-1, add a 1 x 512 x 1024 raw array under that panel's names, copied from the substitute image data if it was loaded, otherwise from the lane's DMA buffer after its first 32 bytes. */
    void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t l1count) override;
private:
    enum {
        m_rawNamesIndex=NamesIndex::BASE,
    };
    std::vector<uint8_t> m_rawBuffer[PGP_MAX_LANES-1]; // Maximum of 7 lanes

    const unsigned m_nAsics = 1; //?
    const unsigned m_nRows = 512;
    const unsigned m_nCols = 1024;
    const unsigned m_nElems = m_nAsics * m_nRows * m_nCols;
    unsigned m_nPanels = 0; // Number of detector panels, also number of lanes since 1 panel/lane
    std::vector<uint16_t> m_substituteRawData; // To load data from LCLS1
    std::vector<std::string> m_panelSerNos;
    std::vector<unsigned> m_segNos; // Must match number of serial numbers if provided!
};

}
