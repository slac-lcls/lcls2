#pragma once

#include "BEBDetector.hh"
#include "psalg/alloc/Allocator.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "xtcdata/xtc/Xtc.hh"

namespace Drp
{

class Jungfrau : public BEBDetector
{
public:
    Jungfrau(Parameters* para, MemPool* pool);

private:
    unsigned _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void _event(XtcData::Xtc&, const void* bufEnd, std::vector<XtcData::Array<uint8_t>>&) override;

private:
    //enum { m_rawNamesIndex=NamesIndex::BASE };
    const unsigned m_nAsics = 1;
    const unsigned m_nRows = 512;
    const unsigned m_nCols = 1024;
    const unsigned m_nElems = m_nAsics * m_nRows * m_nCols;
    unsigned m_nModules = 0;
    std::vector<unsigned> m_segNos;
};

} // namespace Drp
