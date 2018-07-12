#include "psalg/digitizer/Hsd.hh"

#include <stdio.h>
#include <ctype.h>

using namespace Pds::HSD;

Hsd_v1_2_3::Hsd_v1_2_3(Allocator *allocator, const unsigned nChan) // Pass array of lengths nChan
: m_allocator(allocator)
, numPixels(allocator, nChan)
, sPos(allocator, nChan)
, len(allocator, nChan)
, fexPos(allocator, nChan)
, fexPtr(allocator, nChan)
, numFexPeaks(allocator, nChan)
, rawPtr(allocator, nChan)
{
    version = "1.2.3";
    for (unsigned i=0; i<nChan; i++) {
        auto _t = AllocArray1D<uint16_t>(m_allocator, 1600); // FIXME: better way to set array length? Get from hsd configure
        sPos.push_back(_t);
        auto _p = AllocArray1D<uint16_t>(m_allocator, 1600);
        len.push_back(_p);
        auto _r = AllocArray1D<uint16_t>(m_allocator, 1600);
        fexPos.push_back(_r);
        auto _q = AllocArray1D<uint16_t*>(m_allocator, 1600);
        fexPtr.push_back(_q);
    }
}
