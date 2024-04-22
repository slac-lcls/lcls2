#pragma once

#include "BEBDetector.hh"
#include "psalg/alloc/Allocator.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "xtcdata/xtc/Xtc.hh"

namespace Drp
{

class HREncoder : public BEBDetector
{
public:
    HREncoder(Parameters* para, MemPool* pool);

private:
    unsigned _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void _event(XtcData::Xtc&, const void* bufEnd, std::vector<XtcData::Array<uint8_t>>&) override;

private:
    XtcData::NamesId m_evtNamesRaw;
    Heap m_allocator;
};

} // namespace Drp
