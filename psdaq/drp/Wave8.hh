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
                              std::vector< XtcData::Array<uint8_t> >&) override;
private:
    XtcData::NamesId  m_evtNamesRaw;
    XtcData::NamesId  m_evtNamesFex;
    Heap              m_allocator;
};

}
