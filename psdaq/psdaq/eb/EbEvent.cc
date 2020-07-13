#include "EbEvent.hh"
#include "EventBuilder.hh"

#include "xtcdata/xtc/Dgram.hh"

#include <new>
#include <stdlib.h>

using namespace XtcData;
using namespace Pds;
using namespace Pds::Eb;

static const int MaxTimeouts = 50;     // In units of transport.pend() timeouts

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
                 EbEvent*        after,
                 const EbDgram*  cdg,
                 unsigned        prm) :
  t0       (fast_monotonic_clock::now()), // Revisit: Temporary?
  _contract (contract),
  _living   (MaxTimeouts),
  _prm      (prm),
  _damage   (cdg->xtc.damage.value()),
  _last     (_contributions)
{
  *_last++   = cdg;

  _size      = cdg->xtc.sizeofPayload();

  _remaining = contract & ~(1ul << cdg->xtc.src.value());
  if (_remaining == contract)           // Make sure some bit was taken down
  {
    fprintf(stderr, "%s:\n  Source %u isn't in contract %016lx "
            "for %s @ %p, PID %014lx, RoGs %04hx\n",
            __PRETTY_FUNCTION__, cdg->xtc.src.value(), contract,
            TransitionId::name(cdg->service()), cdg, cdg->pulseId(),
            cdg->readoutGroups());
    throw "Fatal: _remaining == contract";
  }

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
**    This function is used to insert a "dummy" contribution into the event.
**    The dummy contribution is identified by the input argument.
**
** --
*/

void EbEvent::_insert(const EbDgram* dummy)
{
  *_last++ = dummy;
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

EbEvent* EbEvent::_add(const EbDgram* cdg)
{
  *_last++   = cdg;

  _size     += cdg->xtc.sizeofPayload();

  uint64_t remaining = _remaining;
  _remaining = remaining & ~(1ul << cdg->xtc.src.value());
  if (_remaining == remaining)          // Make sure some bit was taken down
  {
    fprintf(stderr, "%s:\n  Source %u didn't affect remaining %016lx "
            "for %s @ %p, PID %014lx, RoGs %04hx, contract %016lx\n",
            __PRETTY_FUNCTION__, cdg->xtc.src.value(), remaining,
            TransitionId::name(cdg->service()), cdg, cdg->pulseId(),
            cdg->readoutGroups(), _contract);
    throw "Fatal: _remaining == remaining";
  }

  _damage.increase(cdg->xtc.damage.value());

  _living    = MaxTimeouts;

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
  printf("   Event #%d @ address %p has sequence %014lX\n",
         number, this, sequence());
  printf("    Forward link -> %p, Backward link -> %p\n",
         forward(), reverse());
  printf("    Contributors remaining/requested = %016lX/%016lX\n",
         _remaining, _contract);
  printf("    Total size (in bytes) = %zd\n", _size);

  const EbDgram** const  last    = end();
  const EbDgram*  const* current = begin();
  const EbDgram*         contrib = *current;

  printf("    Creator (%p) was @ source %02x with an environment of 0x%08x\n",
         contrib,
         contrib->xtc.src.value(),
         contrib->env);

  printf("    Contributors to this event:\n");
  while(++current != last)
  {
    contrib = *current;
    printf("     %p: src %02x seq %014lx size %08x env 0x%08x\n",
           contrib,
           contrib->xtc.src.value(),
           contrib->pulseId(),
           contrib->xtc.sizeofPayload(),
           contrib->env);
  }
}
