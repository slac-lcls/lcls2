#pragma once

#include "drp.hh"
#include "XpmDetector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesLookup.hh"

#include <vector>
#include <cstdint>

namespace Drp {

class EpixHRemu : public XpmDetector
{
public:
    EpixHRemu(Parameters* para, MemPool* pool);
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) override;
    unsigned beginrun(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) override;
    void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event) override;
public:
    const unsigned numAsics    = 4;
    const unsigned elemRows    = 144;
    const unsigned elemRowSize = 192;
    const unsigned numElems    = numAsics * elemRows * elemRowSize;
private:
    enum {RawNamesIndex = NamesIndex::BASE, FexNamesIndex};
    std::vector<uint8_t> m_rawBuffer[PGP_MAX_LANES];
};

}
