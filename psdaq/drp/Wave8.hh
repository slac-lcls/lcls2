#pragma once

#include "BEBDetector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "psalg/alloc/Allocator.hh"

namespace Drp {

class Wave8 : public BEBDetector
{
public:
    Wave8(Parameters* para, MemPool* pool);
private:
    unsigned       _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void           _event    (XtcData::Xtc&, const void* bufEnd,
                              uint64_t l1count,
                              std::vector< XtcData::Array<uint8_t> >&) override;
    // For binning into the cube
    virtual void     addToCube(unsigned rawDefIndex, unsigned subIndex, double* dst, XtcData::DescData& rawData) override;
    virtual unsigned rawNamesIndex () override { return EventNamesIndex+0; }
    virtual unsigned cubeNamesIndex() override { return EventNamesIndex+2; }
    virtual XtcData::VarDef rawDef () override;
private:
    XtcData::NamesId  m_evtNamesRaw;
    XtcData::NamesId  m_evtNamesFex;
    Heap              m_allocator;
};

}
