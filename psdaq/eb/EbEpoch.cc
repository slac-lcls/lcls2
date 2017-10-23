/*
** ++
**  Package:
**
**  Abstract:
**
**
**  Author:
**      Michael Huffer, SLAC, (415) 926-4269
**
**  Creation Date:
**	000 - June 1,1998
**
**  Revision History:
**	None.
**
** --
*/

#include "EbEpoch.hh"

using namespace Pds::Eb;

/*
** ++
**
**    Simple debugging tool to format and dump the contents of the object...
**
** --
*/

#include <stdio.h>

void EbEpoch::dump(int number)
{
  printf("  Epoch #%d @ address %p is tagged as %016lX\n",
         number, this, key);
  printf("   Forward link -> %p, Backward link -> %p\n",
         forward(), reverse());

  EbEvent* end   = pending.empty();
  EbEvent* event = pending.forward();

  if(event != end)
  {
    int number = 1;
    do event->dump(number++); while(event = event->forward(), event != end);
  }
  else
    printf("   Epoch has NO pending events...\n");
}
