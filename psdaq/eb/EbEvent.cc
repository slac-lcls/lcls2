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

#include "xtcdata/xtc/Dgram.hh"

#include <new>
#include <stdlib.h>

using namespace XtcData;
using namespace Pds;
using namespace Pds::Eb;

static const int MaxTimeouts = 0x100;      // Revisit: Was 0xffff

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

EbEvent::EbEvent(uint64_t      contract,
                 EventBuilder* eb,
                 EbEvent*      after,
                 const Dgram*  cdg,
                 uint64_t      mask) :
  _sequence(cdg->seq.stamp().pulseId()),
  _contract(contract),
  _eb      (eb),
  _living  (MaxTimeouts),
  _key     (_sequence & mask),
  _last    (_contributions)
{
  const EbContribution* contribution = (EbContribution*)cdg;

  *_last++   = contribution;

  _size      = contribution->payloadSize();

  _remaining = contract & contribution->retire();

  connect(after);
}

/*
** ++
**
**    As soon as an event becomes "complete" its datagram is the only
**    information of value within the event. Therefore, when the event
**    becomes complete it is deleted which cause the destructor to
**    remove the event from the pending queue.
**
** --
*/

Pds::Eb::EbEvent::~EbEvent()
{
}

/*
** ++
**
**    This is the function called by the EB to signify that the event
**    is complete and can be processed.  This function runs in the context
**    of the Back-End task.  The function will simply call the builder's
**    process virtual method for the event and than delete the event.
**
** --
*/

void Pds::Eb::EbEvent::routine()
{
  _eb->process(this);

  delete this;
}

/*
** ++
**
**    This function is used to insert a "dummy" contribution into the event.
**    The dummy contribution is identified by the input argument.
**
** --
*/

void EbEvent::_insert(const Dgram* dummy)
{
  *_last++ = (EbContribution*)dummy;
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

EbEvent* EbEvent::_add(const Dgram* cdg)
{
  const EbContribution* contribution = (EbContribution*)cdg;

  *_last++    = contribution;

  _size      += contribution->payloadSize();

  _remaining &= contribution->retire();

  _living     = MaxTimeouts;

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
  printf("   Event #%d @ address %p has sequence %016lX\n",
         number, this, _sequence);
  printf("    Forward link -> %p, Backward link -> %p\n",
         forward(), reverse());
  printf("    Contributors remaining/requested = %08lX/%08lX\n",
         _remaining, _contract);
  printf("    Total size (in bytes) = %zd\n", _size);

  const EbContribution** const  last    = end();
  const EbContribution*  const* current = begin();
  const EbContribution*         contrib = *current;

  printf("    Creator (%p) was @ source %d with an environment of 0x%lux\n",
         contrib,
         contrib->number(),
         contrib->env);

  printf("    Contributors to this event:\n");
  while(++current != last)
  {
    contrib = *current;
    printf("     src %02x seq %016lx size %08x env 0x%lux\n",
           contrib->number(),
           contrib->seq.stamp().pulseId(),
           contrib->payloadSize(),
           contrib->env);
  }
}
