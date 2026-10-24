/**
 * @file
 * @brief Wave8, the DRP detector class for the Wave8 digitizer (detType wave8 in PGPDetectorApp).
 */
#pragma once

#include "BEBDetector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "psalg/alloc/Allocator.hh"

namespace Drp {

/** BEBDetector for the Wave8 (detType wave8 in PGPDetectorApp), recording raw and FEX data and supporting cube binning. */
class Wave8 : public BEBDetector
{
public:
    /** Construct the BEBDetector base and initialize it with the epics_prefix kwarg. */
    Wave8(Parameters* para, MemPool* pool);
private:
    unsigned       _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void           _event    (XtcData::Xtc&, const void* bufEnd,
                              uint64_t l1count,
                              std::vector< XtcData::Array<uint8_t> >&) override;
    // For binning into the cube
    //    virtual void     addToCube(unsigned rawDefIndex, unsigned valueIndex, unsigned subIndex, double* dst, XtcData::DescData& rawData) override;
    virtual unsigned rawNamesIndex () override { return EventNamesIndex+0; }
    virtual unsigned cubeNamesIndex() override { return EventNamesIndex+2; }
    virtual std::vector<XtcData::VarDef>& rawDef () override;
private:
    XtcData::NamesId  m_evtNamesRaw;
    XtcData::NamesId  m_evtNamesFex;
    Heap              m_allocator;
};

}
