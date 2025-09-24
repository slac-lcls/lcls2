#pragma once

#include "BEBDetector.hh"

namespace Drp {

class TimeTool : public BEBDetector
{
public:
    TimeTool(Parameters* para, MemPool* pool);
private:
    unsigned       _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void           _event    (XtcData::Xtc&, const void* bufEnd,
                              uint64_t l1count,
                              std::vector< XtcData::Array<uint8_t> >&) override;
private:
    XtcData::NamesId  m_evtNamesId;
    unsigned          m_roiLen;
  };

}
