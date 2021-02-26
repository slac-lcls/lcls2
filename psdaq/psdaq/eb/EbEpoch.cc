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

void EbEpoch::dump(int number)
{
  auto now = fast_monotonic_clock::now();
  auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();

  printf(" Epoch #%2d  @ %16p nxt %16p prv %16p key %014lx        age %5ld ms\n",
         number, this, forward(), reverse(), key, age);

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
