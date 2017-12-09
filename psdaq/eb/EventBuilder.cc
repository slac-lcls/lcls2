#include "EventBuilder.hh"
#include "EbEpoch.hh"
#include "EbEvent.hh"
#include "EbContribution.hh"

#include "psdaq/service/Task.hh"
#include "xtcdata/xtc/Dgram.hh"

#include <stdlib.h>

using namespace XtcData;
using namespace Pds;
using namespace Pds::Eb;

EventBuilder::EventBuilder(unsigned epochs,
                           unsigned entries,
                           unsigned sources,
                           uint64_t duration) :
  Timer(),
  _mask(~(duration - 1) & ((1UL << 56) - 1)), // Revisit: ickiness
  _epochFreelist(sizeof(EbEpoch), epochs),
  _eventFreelist(sizeof(EbEvent), epochs * entries),
  _cntrbFreelist(sizeof(EbContribution), epochs * entries * sources),
  _task(new Task(TaskObject("tEB", 100))),
  _duration(100)                        // Timeout rate in ms
{
  if (__builtin_popcountl(duration) != 1)
  {
    fprintf(stderr, "Batch duration (%016lx) must be a power of 2\n",
            duration);
    abort();
  }
}

EventBuilder::~EventBuilder()
{
}

EbEpoch* EventBuilder::_discard(EbEpoch* epoch)
{
  EbEpoch* next = epoch->reverse();
  delete epoch;
  return next;
}

void EventBuilder::_flushBefore(EbEpoch* entry)
{
  EbEpoch* empty = _pending.empty();
  EbEpoch* epoch = entry->reverse();

  while (epoch != empty)
  {
    epoch = epoch->pending.forward() == epoch->pending.empty() ?
      _discard(epoch) :
      epoch->reverse();
  }
}

#include <new>

EbEpoch* EventBuilder::_epoch(uint64_t key, EbEpoch* after)
{
  void* buffer = _epochFreelist.alloc(sizeof(EbEpoch));
  if (!buffer)
  {
    printf("%s: Unable to allocate epoch: key %016lx", __PRETTY_FUNCTION__,
           key);
    printf(" epochFreelist:\n");
    _epochFreelist.dump();
    abort();
  }

  return ::new(buffer) EbEpoch(key, after);
}

EbEpoch* EventBuilder::_match(uint64_t inKey)
{
  EbEpoch*  empty = _pending.empty();
  EbEpoch*  epoch = _pending.reverse();
  uint64_t  key   = inKey & _mask;

  while (epoch != empty)
  {
    uint64_t epochKey = epoch->key;

    if (epochKey == key) return epoch;
    if (epochKey <  key) break;
    epoch = epoch->reverse();
  }

  _flushBefore(epoch);
  return _epoch(key, epoch);
}

EbEvent* EventBuilder::_event(const Dgram* contrib,
                              uint64_t     param,
                              EbEvent*     after)
{
  void* buffer = _eventFreelist.alloc(sizeof(EbEvent));
  if (!buffer)
  {
    printf("%s: Unable to allocate event\n", __PRETTY_FUNCTION__);
    printf("  eventFreelist:\n");
    _eventFreelist.dump();
    abort();
  }

  return ::new(buffer) EbEvent(contract(contrib),
                               _cntrbFreelist,
                               after,
                               contrib,
                               param,
                               _mask);
}

EbEvent* EventBuilder::_insert(EbEpoch*     epoch,
                               const Dgram* contrib,
                               uint64_t     param)
{
  EbEvent* empty = epoch->pending.empty();
  EbEvent* event = epoch->pending.reverse();
  uint64_t key   = contrib->seq.stamp().pulseId();

  while (event != empty)
  {
    uint64_t eventKey = event->sequence();

    if (eventKey == key) return event->_add(contrib, param);
    if (eventKey <  key) break;
    event = event->reverse();
  }

  return _event(contrib, param, event);
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

  delete event;
}

void EventBuilder::_flush(EbEvent* due)
{
  EbEpoch* lastEpoch = _pending.empty();
  EbEpoch* epoch     = _pending.forward();
  EbEvent* last_due  = due;

  do
  {
    EbEvent* lastEvent = epoch->pending.empty();
    EbEvent* event     = epoch->pending.forward();

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
  EbEpoch* epoch = _pending.forward();
  EbEpoch* empty = _pending.empty();

  while (epoch != empty)
  {
    EbEvent* event = epoch->pending.forward();

    if (event != epoch->pending.empty())
    {
      if (!event->_alive())
      {
        printf("Flushing event %014lx, size %zu, remaining %016lx\n",
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
  return _task;
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

void EventBuilder::process(const Dgram* contrib,
                           uint64_t     appParam)
{
  EbEpoch* epoch = _match(contrib->seq.stamp().pulseId());
  EbEvent* event = _insert(epoch, contrib, appParam);
  if (!event->_remaining)  _flush(event);
}

unsigned EventBuilder::processBulk(const Dgram* contrib,
                                   uint64_t     appParam)
{
  unsigned cnt   = 0;
  EbEpoch* epoch = _match(contrib->seq.stamp().pulseId());
  EbEvent* event;

  const Dgram*  next = (Dgram*)contrib->xtc.payload();
  const Dgram*  last = (Dgram*)contrib->xtc.next();
  while(next != last)
  {
    //if (lverbose > 1)                 // Revisit: lverbose is not defined
    //{
    //  unsigned from = next->xtc.src.log() & 0xff;
    //  printf("EventBuilder found          a  contrib @ %16p, ts %014lx, sz %3zd from Contrib %d\n",
    //         next, next->seq.stamp().pulseId(), sizeof(*next) + next->xtc.sizeofPayload(), from);
    //}

    event = _insert(epoch, next, appParam);

    ++cnt;

    next = (Dgram*)next->xtc.next();
  }

  _flush(event);

  return cnt;
}

/*
** ++
**
**
** --
*/

void EventBuilder::dump(unsigned detail)
{
  if (detail)
  {
    EbEpoch* last  = _pending.empty();
    EbEpoch* epoch = _pending.forward();

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

  printf("\nEvent Builder contribution pool:\n");
  _cntrbFreelist.dump();
}
