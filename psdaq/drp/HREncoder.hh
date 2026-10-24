/**
 * @file
 * @brief HREncoder, the DRP detector class for the high-rate encoder.
 */
#pragma once

#include "BEBDetector.hh"
#include "psalg/alloc/Allocator.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "xtcdata/xtc/Xtc.hh"

namespace Drp
{

/** BEBDetector for the high-rate encoder (detType hrencoder in PGPDetectorApp). */
class HREncoder : public BEBDetector
{
public:
    /** Initialize the BEBDetector with the epics_prefix kwarg; a timebase kwarg of 119M sets the extra event-batcher level. */
    HREncoder(Parameters* para, MemPool* pool);

private:
    unsigned _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void _event(XtcData::Xtc&, const void* bufEnd, uint64_t l1count, std::vector<XtcData::Array<uint8_t>>&) override;

private:
    XtcData::NamesId m_evtNamesRaw;
    Heap m_allocator;
};

} // namespace Drp
