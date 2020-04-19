#pragma once

#include "BEBDetector.hh"

namespace Drp {

class TimeTool : public BEBDetector
{
public:
    TimeTool(Parameters* para, MemPool* pool);
private:
    unsigned       _configure(XtcData::Xtc&) override;
    void           _event    (XtcData::Xtc&,
                              std::vector< XtcData::Array<uint8_t> >&) override;
private:
    XtcData::NamesId  m_evtNamesId;
  };

}
