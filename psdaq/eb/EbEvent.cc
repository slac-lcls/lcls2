/*
** ++
**  Package:
**      Eb
**
**  Abstract:
**      Non-inline functions for class "EbEvent.hh"
**
**  Author:
**      Michael Huffer, SLAC, (650) 926-4269
**
**  Creation Date:
**	000 - June 1,1998
**
**  Revision History:
**	None.
**
** --
*/

#include "EbEvent.hh"
#include "EbContribution.hh"
#include "EventBuilder.hh"

#include "psdaq/service/SysClk.hh"

using namespace Pds::Eb;

static const int MaxTimeouts=0x100;     // Revisit: Was 0xffff

// Revisit: Fix stale comments:
/*
** ++
**
**   Constructor. The creation of an event is initiated by the event builder,
**   whose identity is passed and saved by the event ("eb"). The builder
**   also provides the set of participants for this event ("contract"),
**   the packet which initiated the construction of the event ("packet"),
**   and the location in the pending queue ("after") at which the event
**   should insert itself. the packet will be used to determine the
**   sequence number of the event. Note, that the event keeps a (single)
**   linked list of each packet contribution. This list doubles as the
**   DMA list passed to the Universe's DMA engine.
**
** --
*/

EbEvent::EbEvent(uint64_t        contract,
                 EventBuilder*   eb,
                 EbEvent*        after,
                 EbContribution* contrib,
                 uint64_t        mask,
                 void*           data) :
  _data(data)
{
  // Make sure we didn't run out of heap before initializing
  if (!this)  return;

  ClockTime key = contrib->seq.clock();
  _sequence     = key;
  _key          = key.u64() & mask;
  _eb           = eb;
  _living       = MaxTimeouts;
  _tail         = _pending;
  _head         = _pending;

  *_head++      = contrib;

  unsigned size = contrib->payloadSize();

  _size = size;

  _contract  = contract;
  _remaining = contract & contrib->retire();

  connect(after);
}

/*
** ++
**
**    This function is used to insert a "dummy" contribution to the event.
**    The dummy contribution is identified by the input argument.
**
** --
*/

void EbEvent::_insert(EbContribution* dummy)
{
  *_head++ = dummy;
}

/*
** ++
**
**    This function is used to ADD a new contribution to the event. The
**    contribution is identified by the input argument. The function will
**    refresh its living counter (see "isAlive"), add the contribution
**    onto the DMA list, initializing the part of the packet which will
**    be used by the DMA engine (principally, its size and VME address),
**    and finally, mark off the contribution from its "remaining" list.
**    A pointer to the object itself is returned.
**
** --
*/

EbEvent* EbEvent::_add(EbContribution* contrib)
{
  *_head++ = contrib;

  unsigned size = contrib->payloadSize();

  _size += size;

  _remaining &= contrib->retire();

  return this;
}

/*
** ++
**
**    Simple debugging tool to format and dump the contents of the object...
**
** --
*/

#include <stdio.h>

void EbEvent::dump(int number)
{
  printf("   Event #%d @ address %08X has sequence %08X%08X\n",
         number, (int)this,
         _sequence.seconds(), _sequence.nanoseconds());
  printf("    Forward link -> %08X, Backward link -> %08X\n",
         (unsigned)forward(), (unsigned)reverse());
  printf("    Contributors remaining/requested = %08X/%08X\n",
         _remaining, _contract);
  printf("    Total size (in bytes) = %d\n", _size);

  EbContribution** next  = _tail;
  EbContribution** empty = _head;
  EbContribution*  contrib = *next++;

  printf("    Creator(%p) was @ source %d with an environment of %08X\n",
         contrib,
         contrib->number(),
         contrib->_contribution->env.value());

  printf("    Contribs for this event:\n");
  while(next != empty)
  {
    printf("src %02x seq %08x/%08x size %08x env %08x\n",
           contrib->number(),
           contrib->seq.clock().seconds(),
           contrib->seq.clock().nanoseconds(),
           contrib->payloadSize(),
           contrib->_contribution->env.value());
    contrib = *next++;
  }
}
