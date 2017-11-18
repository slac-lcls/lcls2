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

#include <stdlib.h>

using namespace XtcData;
using namespace Pds;
using namespace Pds::Eb;

static const int MaxTimeouts = 100;      // Revisit: Was 0xffff

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
                 uint64_t      prm,
                 uint64_t      mask) :
  _pending()
{
  uint64_t key = cdg->seq.stamp().pulseId();
  _sequence = key;
  _key      = key & mask;
  _eb       = eb;
  _living   = MaxTimeouts;

  EbContribution* contrib = _contribution(cdg, prm);

  _size      = contrib->payloadSize();

  _contract  = contract;
  _remaining = contract & contrib->retire();

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
  EbContribution* empty   = _pending.empty();
  EbContribution* contrib = _pending.reverse();

  while (contrib != empty)
  {
    EbContribution* next = contrib->reverse();
    delete contrib;
    contrib = next;
  }
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
  _pending.insert(dummy);
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

EbEvent* EbEvent::_add(const Dgram* cdg,
                       uint64_t     prm)
{
  EbContribution* contrib = _contribution(cdg, prm);

  unsigned size = contrib->payloadSize();

  _size += size;

  _remaining &= contrib->retire();

  _living = MaxTimeouts;

  return this;
}

EbContribution* EbEvent::_contribution(const Dgram* cdg,
                                       uint64_t     prm)
{
  EbContribution* contrib = _pending.reverse();

  contrib = new(&_eb->_cntrbFreelist) EbContribution(cdg, prm, contrib);
  if (!contrib)
  {
    printf("%s: Unable to allocate contribution\n", __PRETTY_FUNCTION__);
    printf("  cntrbFreelist:\n");
    _eb->_cntrbFreelist.dump();
    abort();
  }

  return contrib;
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

  EbContribution* last    = _pending.empty();
  EbContribution* contrib = _pending.forward();

  printf("    Creator (%p) was @ source %d with an environment of %08X\n",
         contrib,
         contrib->number(),
         contrib->datagram()->env.value());

  printf("    Contributors to this event:\n");
  while((contrib = contrib->forward()) != last)
  {
    printf("     src %02x seq %016lx size %08x env %08x\n",
           contrib->number(),
           contrib->datagram()->seq.stamp().pulseId(),
           contrib->payloadSize(),
           contrib->datagram()->env.value());
  }
}
