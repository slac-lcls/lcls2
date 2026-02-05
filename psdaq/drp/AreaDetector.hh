#pragma once

#include "drp.hh"
#include "XpmDetector.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesLookup.hh"

namespace Drp {

class AreaDetector : public XpmDetector
{
public:
    AreaDetector(Parameters* para, MemPool* pool);
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) override;
    unsigned beginrun(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) override;
    // Avoid "overloaded virtual function "Drp::Detector::event" is only partially overridden" warning
    using Detector::event;
    void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t l1count) override;
    // For binning into the cube
    virtual void     addToCube(unsigned rawDefIndex, double* dst, XtcData::DescData& rawData) override;
    virtual unsigned rawNamesIndex () override { return RawNamesIndex; }
    virtual unsigned cubeNamesIndex() override { return CubeNamesIndex; }
    virtual XtcData::VarDef rawDef () override;
private:
    enum {RawNamesIndex = NamesIndex::BASE, FexNamesIndex, CubeNamesIndex};
    enum {Pedestals, Gains, NumConstants};
    std::vector<char*> m_constants;
};

}
