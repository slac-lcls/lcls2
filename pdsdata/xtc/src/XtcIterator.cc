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

#include "pdsdata/xtc/XtcIterator.hh"
#include "pdsdata/xtc/Xtc.hh"

using namespace Pds;

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

void XtcIterator::iterate(Xtc* root) 
{
  if (root->damage.value() & ( 1 << Damage::IncompleteContribution))
    return;

  Xtc* xtc     = (Xtc*)root->payload();
  int remaining = root->sizeofPayload();

  while(remaining > 0)
  {
    if(xtc->extent==0) break; // try to skip corrupt event
    if(!process(xtc)) break;
    remaining -= xtc->sizeofPayload() + sizeof(Xtc);
    xtc      = xtc->next();
  }

  return;
}
