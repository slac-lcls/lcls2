#include "psalg/include/hsd.hh"

#include <stdio.h>
#include <ctype.h>

using namespace Pds::HSD;

Hsd_v1_0_0::Hsd_v1_0_0(Allocator *allocator, const unsigned nChan)
: m_allocator(allocator)
, numPixels(allocator, nChan)
, sPosx(allocator, nChan)
, lenx(allocator, nChan)
, fexPos(allocator, nChan)
, fexPtr(allocator, nChan)
, numFexPeaksx(allocator, nChan)
, rawPtr(allocator, nChan)
{
    version = "1.0.0";
    for (unsigned i=0; i<nChan; i++) {
        auto _t = AllocArray1D<uint16_t>(m_allocator, 1600); // FIXME: better way to set array length?
        sPosx.push_back(_t);
        auto _p = AllocArray1D<uint16_t>(m_allocator, 1600);
        lenx.push_back(_p);
        auto _r = AllocArray1D<uint16_t>(m_allocator, 1600);
        fexPos.push_back(_r);
        auto _q = AllocArray1D<uint16_t*>(m_allocator, 1600);
        fexPtr.push_back(_q);
    }
}
