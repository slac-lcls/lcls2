/*
 ** ++
 **  Package:
 **	OdfContainer
 **
 **  Abstract:
 **     non-inline functions for "InXtcIterator.hh"
 **
 **  Author:
 **      Michael Huffer, SLAC, (415) 926-4269
 **
 **  Creation Date:
 **	000 - October 11,1998
 **
 **  Revision History:
 **	None.
 **
 ** --
 */

#include "XtcIterator.hh"
#include "Xtc.hh"

#define UNLIKELY(expr)  __builtin_expect(!!(expr), 0)

using namespace XtcData;

/*
 ** ++
 **
 **   Iterate over the collection specifed as an argument to the function.
 **   For each "Xtc" found call back the "process" function. If the
 **   "process" function returns zero (0) the iteration is aborted and
 **   control is returned to the caller. Otherwise, control is returned
 **   when all elements of the collection have been scanned.
 **
 ** --
 */

void XtcIterator::iterate(Xtc* root, const void* bufEnd)
{
    if (root->damage.value() & (1 << Damage::Corrupted)) return;

    Xtc* xtc = (Xtc*)root->payload();
    int remaining = root->sizeofPayload();

    while (remaining > 0) {
        if (bufEnd && UNLIKELY(xtc >= (Xtc*)bufEnd)) {
            // protect against buffer overrun
            printf("*** %s:%d: corrupt xtc, would overrun buffer\n",__FILE__,__LINE__);
            abort();
        }
        if (xtc->extent < sizeof(Xtc)) {
            printf("*** %s:%d: corrupt xtc with too small extent: %d\n",__FILE__,__LINE__,xtc->extent);
            abort();
        }
        if (!process(xtc, bufEnd)) break;
        remaining -= xtc->sizeofPayload() + sizeof(Xtc);
        xtc = xtc->next();
    }

    return;
}
