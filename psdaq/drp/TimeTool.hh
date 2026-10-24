/**
 * @file
 * @brief TimeTool, a BEBDetector for timetool data (detType tt in PGPDetectorApp).
 */
#pragma once

#include "BEBDetector.hh"

namespace Drp {

/** BEBDetector for timetool data (detType tt in PGPDetectorApp). */
class TimeTool : public BEBDetector
{
public:
    /** Construct the BEBDetector base and initialize it with _init(detName) and _init_feb(). */
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
