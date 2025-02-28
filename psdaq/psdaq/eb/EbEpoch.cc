#include <chrono>

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

void EbEpoch::dump(unsigned detail, int number)
{
  if (!detail)
    printf(" Epoch #%2d  key %014lx\n",
           number, key);
  else
    printf(" Epoch #%2d  @ %16p nxt %16p prv %16p key %014lx\n",
           number, this, forward(), reverse(), key);

  EbEvent* end   = pending.empty();
  EbEvent* event = pending.forward();

  if(event != end)
  {
    if (detail)
    {
      int number = 1;
      do event->dump(detail, number++); while(event = event->forward(), event != end);
    }
    else
      event->dump(detail, 0);
  }
  else
    printf("   Epoch has NO pending events...\n");
}
