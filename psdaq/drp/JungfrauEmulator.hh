#pragma once

#include "drp.hh"
#include "XpmDetector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesLookup.hh"

#include <vector>
#include <cstdint>

namespace Drp {

class JungfrauEmulator : public XpmDetector
{
public:
    JungfrauEmulator(Parameters* para, MemPool* pool);
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) override;
    unsigned beginrun(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) override;
    void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event) override;
private:
    enum {
        m_rawNamesIndex=NamesIndex::BASE,
        m_fexNamesIndex // Unused... for now...
    };
    std::vector<uint8_t> m_rawBuffer[PGP_MAX_LANES-1]; // Maximum of 7 lanes

    const unsigned m_nAsics = 1; //?
    const unsigned m_nRows = 512;
    const unsigned m_nCols = 1024;
    const unsigned m_nElems = m_nAsics * m_nRows * m_nCols;
    unsigned m_nPanels = 0; // Number of detector panels, also number of lanes since 1 panel/lane
    std::vector<uint16_t> m_substituteRawData; // To load data from LCLS1
};

}
