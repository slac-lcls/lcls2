#include "EventBuilder.hh"
#include "EbEpoch.hh"
#include "EbEvent.hh"
#include "EbContribution.hh"

#include "psdaq/service/Task.hh"
#include "xtcdata/xtc/Dgram.hh"

#include <stdlib.h>
#include <new>

using namespace XtcData;
using namespace Pds;
using namespace Pds::Eb;

EventBuilder::EventBuilder(unsigned epochs,
                           unsigned entries,
                           unsigned sources,
                           uint64_t duration) :
  Timer(),
  _mask(~(duration - 1) & ((1UL << PulseId::NumPulseIdBits) - 1)),
  _epochFreelist(epochs),
  _eventFreelist(epochs * entries, sources * sizeof(Dgram*)),
  _timerTask(new Task(TaskObject("tEB_Timeout"))),
  _duration(100)                        // Timeout rate in ms
{
  if (duration & (duration - 1))
  {
    fprintf(stderr, "%s: Batch duration (%016lx) must be a power of 2\n",
            __func__, duration);
    abort();
  }
}

EventBuilder::~EventBuilder()
{
  printf("EbEpoch freelist:\n");
  _epochFreelist.dump();

  printf("EbEvent freelist:\n");
  _eventFreelist.dump();

  _timerTask->destroy();
}

EbEpoch* EventBuilder::_discard(EbEpoch* epoch)
{
  EbEpoch* next = epoch->reverse();
  _epochFreelist.free(epoch->disconnect());
  return next;
}

void EventBuilder::_flushBefore(EbEpoch* entry)
{
  const EbEpoch* const empty = _pending.empty();
  EbEpoch*             epoch = entry->reverse();

  while (epoch != empty)
  {
    epoch = epoch->pending.forward() == epoch->pending.empty() ?
      _discard(epoch) :
      epoch->reverse();
  }
}

EbEpoch* EventBuilder::_epoch(uint64_t key, EbEpoch* after)
{
  void* buffer = _epochFreelist.allocate(key);
  if (buffer)  return ::new(buffer) EbEpoch(key, after);

  fprintf(stderr, "%s: Unable to allocate epoch: key %016lx", __PRETTY_FUNCTION__,
          key);
  printf(" epochFreelist:\n");
  _epochFreelist.dump();
  abort();
}

EbEpoch* EventBuilder::_match(uint64_t inKey)
{
  const uint64_t key = (inKey & _mask) >> __builtin_ctzl(_mask);

  if (_epochFreelist.isAllocated(key))
  {
    EbEpoch* epoch = &_epochFreelist[key];
    if (epoch->key == key)  return epoch;

    fprintf(stderr, "%s: Epoch has unexpected sequence number 0x%014lx: expected 0x%014lx\n",
            __PRETTY_FUNCTION__, epoch->key, key);
    _epochFreelist.dump();
    abort();
  }

  const EbEpoch* const empty = _pending.empty();
  EbEpoch*             epoch = _pending.reverse();

  while (epoch != empty)
  {
    if (epoch->key < key) break;
    epoch = epoch->reverse();
  }

  _flushBefore(epoch);
  return _epoch(key, epoch);
}

EbEvent* EventBuilder::_event(const Dgram* ctrb,
                              EbEvent*     after)
{
  void* buffer = _eventFreelist.allocate(ctrb->seq.pulseId().value());
  if (buffer)  return ::new(buffer) EbEvent(contract(ctrb),
                                            this,
                                            after,
                                            ctrb);

  fprintf(stderr, "%s: Unable to allocate event\n", __PRETTY_FUNCTION__);
  printf("  eventFreelist:\n");
  _eventFreelist.dump();
  abort();
}

EbEvent* EventBuilder::_insert(EbEpoch*     epoch,
                               const Dgram* ctrb)
{
  const uint64_t key = ctrb->seq.pulseId().value();
  if (_eventFreelist.isAllocated(key))
  {
    EbEvent* event = &_eventFreelist[key];
    if (event->sequence() == key)  return event->_add(ctrb);

    fprintf(stderr, "%s: Attempt to add contribution with seq 0x%014lx to event 0x%014lx\n",
            __PRETTY_FUNCTION__, key, event->sequence());
    abort();
  }

  const EbEvent* const empty = epoch->pending.empty();
  EbEvent*             event = epoch->pending.reverse();

  while (event != empty)
  {
    if (event->sequence() < key) break;
    event = event->reverse();
  }

  return _event(ctrb, event);
}

EbEvent* EventBuilder::_insert(EbEpoch*     epoch,
                               const Dgram* ctrb,
                               EbEvent*     event)
{
  const uint64_t key = ctrb->seq.pulseId().value();

  if (_eventFreelist.isAllocated(key))
  {
    EbEvent& event = _eventFreelist[key];
    if (event.sequence() == key)  return event._add(ctrb);

    fprintf(stderr, "%s: Attempt to add contribution with seq 0x%014lx to event 0x%014lx\n",
            __PRETTY_FUNCTION__, key, event.sequence());
    _eventFreelist.dump();
    abort();
  }

  const EbEvent* const empty    = epoch->pending.empty();
  EbEvent*             after    = event;
  bool                 reversed = false;

  while (event != empty)
  {
    if (key > event->sequence())
    {
      if (reversed)  break;
      after    = event;
      event    = event->forward();
    }
    else
    {
      event    = event->reverse();
      after    = event;
      reversed = true;
    }
  }

  return _event(ctrb, after);
}

void EventBuilder::_fixup(EbEvent* event) // Always called with remaining != 0
{
  uint64_t& remaining = event->_remaining;

  do
  {
    unsigned srcId = __builtin_ffsl(remaining) - 1;

    fixup(event, srcId);

    remaining &= ~(1ul << srcId);
  }
  while (remaining);
}

void EventBuilder::_retire(EbEvent* event)
{
  event->disconnect();

  process(event);

  _eventFreelist.free(event);
}

void EventBuilder::_flush(EbEvent* due)
{
  const EbEpoch* const lastEpoch = _pending.empty();
  EbEpoch*             epoch     = _pending.forward();
  const EbEvent*       last_due  = due;

  do
  {
    const EbEvent* const lastEvent = epoch->pending.empty();
    EbEvent*             event     = epoch->pending.forward();

    while (event != lastEvent)
    {
      if (event->_remaining)  return;
      if (event == last_due)
      {
        _retire(event);

        return;
      }

      EbEvent* next = event->forward();

      _retire(event);

      event = next;
    }
  }
  while (epoch = epoch->forward(), epoch != lastEpoch);
}

void EventBuilder::expired()            // Periodically called from a timer
{
  EbEpoch*             epoch = _pending.forward();
  const EbEpoch* const empty = _pending.empty();

  while (epoch != empty)
  {
    EbEvent*             event = epoch->pending.forward();
    const EbEvent* const last  = epoch->pending.empty();

    if (event != last)
    {
      if (event->_remaining && !event->_alive())
      {
        printf("Event timed out: %014lx, size %zu, remaining %016lx\n",
               event->sequence(),
               event->size(),
               event->_remaining);

        if (event->_remaining) _fixup(event);

        _flush(event);
      }
      return; // Revisit: This seems wrong, but is original.
              //          Seems like all expired events should be flushed,
              //          not just one per timeout cycle, but it was maybe
              //          done this way to allow events backed up behind the
              //          one that's just been timed out to complete normally
    }

    epoch = epoch->forward();
  }
}

Task* EventBuilder::task()
{
  return _timerTask;
}

unsigned EventBuilder::duration() const
{
  return _duration;
}

unsigned EventBuilder::repetitive() const
{
  return 1;
}

/*
** ++
**
**    This method receives event contributions and starts the event building
**    process.
**
**    Although contributions are ostensibly collected and posted in time order,
**    it seems like it might be possible for the machine on which this event
**    builder is running to notify this process that a new contribution has
**    arrived slightly out of time order.  Thus, a contribution that would
**    complete an older event might already be on the machine with the
**    notification stuck in the transport completion queue until the current
**    contribution (and possibly a few others) has been processed.  Since we
**    don't want to unnecessarily penalize incomplete events, the _flush()
**    below will return as soon as it finds an incomplete event, despite the
**    current event being complete.  Presumably a subsequent call to process()
**    will complete such events well before the timeout is invoked, and the
**    flush will proceed to deliver all thus far completed events to the
**    appication in time order.  Only those events that time out will be
**    fixed up to force their completion, and still delivered in order.
**
** --
*/

void EventBuilder::process(const Dgram* ctrb)
{
  if (ctrb->seq.isBatch())
  {
    _processBulk(ctrb);
  }
  else
  {
    EbEpoch* epoch = _match(ctrb->seq.pulseId().value());

    //if (lverbose > 1)                 // Revisit: lverbose is not defined
    //{
    //  unsigned from = ctrb->xtc.src.log() & 0xff;
    //  printf("EventBuilder found          a  ctrb          @ %16p, pid %014lx, sz %4zd from Ctrb %d\n",
    //         ctrb, ctrb->seq.pulseId().value(), sizeof(*ctrb) + ctrb->xtc.sizeofPayload(), from);
    //}

    EbEvent* event = _insert(epoch, ctrb);
    if (!event->_remaining)  _flush(event);
  }
}

void EventBuilder::_processBulk(const Dgram* ctrb)
{
  EbEpoch* epoch = _match(ctrb->seq.pulseId().value());
  EbEvent* event = epoch->pending.forward();

  const Dgram* next = (Dgram*)ctrb->xtc.payload();
  const Dgram* last = (Dgram*)ctrb->xtc.next();
  while (next != last)
  {
    //if (lverbose > 1)                 // Revisit: lverbose is not defined
    //{
    //  unsigned from = next->xtc.src.log() & 0xff;
    //  printf("EventBuilder found          a  ctrb          @ %16p, pid %014lx, sz %4zd from Ctrb %d\n",
    //         next, next->seq.pulseId().value(), sizeof(*next) + next->xtc.sizeofPayload(), from);
    //}

    event = _insert(epoch, next, event);

    next = (Dgram*)next->xtc.next();
  }

  if (!event->_remaining)  _flush(event);
}

/*
** ++
**
**
** --
*/

void EventBuilder::dump(unsigned detail) const
{
  if (detail)
  {
    const EbEpoch* const last  = _pending.empty();
    EbEpoch*             epoch = _pending.forward();

    if (epoch != last)
    {
      int number = 1;
      do epoch->dump(number++); while (epoch = epoch->forward(), epoch != last);
    }
    else
      printf(" Event Builder has no pending events...\n");
  }

  printf("\nEvent Builder epoch pool:\n");
  _epochFreelist.dump();

  printf("\nEvent Builder event pool:\n");
  _eventFreelist.dump();
}
