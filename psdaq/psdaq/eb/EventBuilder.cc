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
                           uint64_t duration,
                           unsigned verbose) :
  Timer(),
  _mask(PulseId(~(duration - 1), 0).value()),
  _epochFreelist(sizeof(EbEpoch), epochs),
  _epochLut(epochs),
  _eventFreelist(sizeof(EbEvent) + sources * sizeof(Dgram*), epochs * entries),
  _eventLut(epochs * entries),
  _verbose(verbose),
  _duration(100),                       // Timeout rate in ms
  _timerTask(new Task(TaskObject("tEB_Timeout")))
{
  if (duration & (duration - 1))
  {
    fprintf(stderr, "%s: Epoch duration (%016lx) must be a power of 2\n",
            __func__, duration);
    abort();
  }
  // Revisit: For now we punt on the power-of-2 requirement in order to
  //          accomodate transitions, especially in the MEB case
  //if (epochs & (epochs - 1))
  //{
  //  fprintf(stderr, "%s: Number of epochs (%08x) must be a power of 2\n",
  //          __func__, epochs);
  //  abort();
  //}
  //if (entries & (entries - 1))
  //{
  //  fprintf(stderr, "%s: Number of entries per epoch (%08x) must be a power of 2\n",
  //          __func__, entries);
  //  abort();
  //}
}

EventBuilder::~EventBuilder()
{
  _timerTask->destroy();
}

unsigned EventBuilder::_epIndex(uint64_t key) const
{
  //return (key >> __builtin_ctzl(_mask)) & (_epochLut.size() - 1);
  return (key >> __builtin_ctzl(_mask)) % _epochLut.size();
}

unsigned EventBuilder::_evIndex(uint64_t key) const
{
  //return key & (_eventLut.size() - 1);
  return key % _eventLut.size();
}

EbEpoch* EventBuilder::_discard(EbEpoch* epoch)
{
  EbEpoch* next = epoch->reverse();

  const uint64_t key   = epoch->key;
  EbEpoch*&      entry = _epochLut[_epIndex(key)];
  if (entry && (entry->key == key))  entry = nullptr;

  delete epoch;

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
  void* buffer = _epochFreelist.alloc(sizeof(EbEpoch));
  if (buffer)
  {
    EbEpoch*  epoch = ::new(buffer) EbEpoch(key, after);
    unsigned  index = _epIndex(key);
    EbEpoch*& entry = _epochLut[index];
    if (!entry)  entry = epoch;
    //else { printf("Epoch list entry %p is already allocated with key %014lx\n", entry, entry->key);
    //  printf("New epoch %p pid %014lx, index %d, shift %zd\n",
    //         epoch, epoch->key, index, _epochLut.size());
    //  entry->dump(0);
    //}
    return epoch;
  }

  fprintf(stderr, "%s:\n  Unable to allocate epoch: key %014lx\n",
          __PRETTY_FUNCTION__, key);
  printf(" epochFreelist:\n");
  _epochFreelist.dump();
  abort();
}

EbEpoch* EventBuilder::_match(uint64_t inKey)
{
  const uint64_t key = inKey & _mask;

  EbEpoch* epoch = _epochLut[_epIndex(key)];
  if (epoch && (epoch->key == key))  return epoch;

  const EbEpoch* const empty = _pending.empty();
  epoch                      = _pending.reverse();

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

EbEvent* EventBuilder::_event(const Dgram* ctrb,
                              EbEvent*     after,
                              unsigned     prm)
{
  void* buffer = _eventFreelist.alloc(sizeof(EbEvent));
  if (buffer)
  {
    EbEvent*  event = ::new(buffer) EbEvent(contract(ctrb),
                                            after,
                                            ctrb,
                                            prm);
    unsigned  index = _evIndex(ctrb->seq.pulseId().value());
    EbEvent*& entry = _eventLut[index];
    if (!entry)  entry = event;
    //else { printf("Event list entry %p is already allocated with key %014lx\n", entry, entry->sequence());
    //  printf("New event %p pid %014lx, index %d, mask %08lx, shift %zd\n",
    //         event, event->sequence(), index, _mask, _eventLut.size());
    //  entry->dump(0);
    //}
    return event;
  }

  fprintf(stderr, "%s:\n  Unable to allocate event\n", __PRETTY_FUNCTION__);
  printf("  eventFreelist:\n");
  _eventFreelist.dump();
  abort();
}

EbEvent* EventBuilder::_insert(EbEpoch*     epoch,
                               const Dgram* ctrb,
                               EbEvent*     after,
                               unsigned     prm)
{
  const uint64_t key = ctrb->seq.pulseId().value();

  EbEvent* event = _eventLut[_evIndex(key)];
  if (event && (event->sequence() == key))  return event->_add(ctrb);

  bool                 reversed = false;
  const EbEvent* const empty    = epoch->pending.empty();
  event                         = after;

  while (event != empty)
  {
    const uint64_t eventKey = event->sequence();

    if (key == eventKey) return event->_add(ctrb);
    if (key >  eventKey)
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

  return _event(ctrb, after, prm);
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

  const uint64_t key   = event->sequence();
  EbEvent*&      entry = _eventLut[_evIndex(key)];
  if (entry && (entry->sequence() == key))  entry = nullptr;

  delete event;
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

void EventBuilder::process(const Dgram* ctrb, unsigned prm)
{
  EbEpoch* epoch = _match(ctrb->seq.pulseId().value());
  EbEvent* event = epoch->pending.forward();
  EbEvent* due   = nullptr;

  while (true)
  {
    if (_verbose > 1)
    {
      uint64_t pid = ctrb->seq.pulseId().value();
      unsigned ctl = ctrb->seq.pulseId().control();
      size_t   sz  = sizeof(*ctrb) + ctrb->xtc.sizeofPayload();
      unsigned src = ctrb->xtc.src.value();
      printf("EB found          a  ctrb                 @ "
             "%16p, ctl %02x, pid %014lx, sz %4zd, src %2d, parm %08x\n",
             ctrb, ctl, pid, sz, src, prm);
    }

    event = _insert(epoch, ctrb, event, prm);
    if (!event->_remaining)  due = event;

    if (!ctrb->seq.isBatch())  break;

    ctrb = reinterpret_cast<const Dgram*>(ctrb->xtc.next());
  }

  if (due)  _flush(due);
}

/*
** ++
**
**
** --
*/

void EventBuilder::dump(unsigned detail) const
{
  printf("\nEvent builder dump:\n");

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

  printf("Event Builder epoch pool:\n");
  _epochFreelist.dump();

  printf("Event Builder event pool:\n");
  _eventFreelist.dump();
}
