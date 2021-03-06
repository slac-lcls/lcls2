#include "eb.hh"
#include "EventBuilder.hh"
#include "EbEpoch.hh"
#include "EbEvent.hh"

#include "psdaq/service/Task.hh"
#include "xtcdata/xtc/Dgram.hh"

#include <stdlib.h>
#include <new>
#include <chrono>

using namespace XtcData;
using namespace Pds;
using namespace Pds::Eb;

static constexpr unsigned CLS = 64;     // Cache Line Size


EventBuilder::EventBuilder(unsigned        epochs,
                           unsigned        entries,
                           unsigned        sources,
                           uint64_t        duration,
                           unsigned        timeout,
                           const unsigned& verbose) :
  _pending(),
  _mask(~PulseId(duration - 1).pulseId()),
  _epochFreelist(sizeof(EbEpoch), epochs, CLS),
  _epochLut(epochs),
  _eventFreelist(sizeof(EbEvent) + sources * sizeof(EbDgram*), epochs * entries, CLS),
  _eventLut(epochs * entries),
  _eventTimeout(timeout),
  _tmoEvtCnt(0),
  _fixupCnt(0),
  _missing(0),
  _verbose(verbose)
{
  if (duration & (duration - 1))
  {
    fprintf(stderr, "%s:\n  Epoch duration (%016lx) must be a power of 2\n",
            __func__, duration);
    throw "Epoch duration must be a power of 2";
  }
  // Revisit: For now we punt on the power-of-2 requirement in order to
  //          accomodate transitions, especially in the MEB case
  //if (epochs & (epochs - 1))
  //{
  //  fprintf(stderr, "%s:\n  Number of epochs (%08x) must be a power of 2\n",
  //          __func__, epochs);
  //  throw "Number of epochs must be a power of 2";
  //}
  //if (entries & (entries - 1))
  //{
  //  fprintf(stderr, "%s:\n  Number of entries per epoch (%08x) must be a power of 2\n",
  //          __func__, entries);
  //  throw "Number of entries per epoch must be a power of 2";
  //}
}

EventBuilder::~EventBuilder()
{
}

void EventBuilder::resetCounters()
{
  _eventFreelist.clearCounters();
  _epochFreelist.clearCounters();
  _tmoEvtCnt = 0;
  _fixupCnt  = 0;
  _missing   = 0;
}

void EventBuilder::clear()
{
  const EbEpoch* const lastEpoch = _pending.empty();
  EbEpoch*             epoch     = _pending.forward();

  while (epoch != lastEpoch)
  {
    const EbEvent* const lastEvent = epoch->pending.empty();
    EbEvent*             event     = epoch->pending.forward();

    while (event != lastEvent)
    {
      EbEvent* next = event->forward();

      event->disconnect();

      _eventLut[_evIndex(event->sequence())] = nullptr;

      delete event;

      event = next;
    }
    epoch = epoch->forward();
  }

  _flushBefore(_pending.reverse());

  resetCounters();

  for (auto it = _epochLut.begin(); it != _epochLut.end(); ++it)
    *it = nullptr;
  for (auto it = _eventLut.begin(); it != _eventLut.end(); ++it)
    *it = nullptr;
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
  if (entry == epoch)  entry = nullptr;

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
    //  printf("New epoch %p pid %014lx, index %u, shift %zd\n",
    //         epoch, epoch->key, index, _epochLut.size());
    //  entry->dump(0);
    //}
    return epoch;
  }

  fprintf(stderr, "%s:\n  Unable to allocate epoch: key %014lx\n",
          __PRETTY_FUNCTION__, key);
  printf(" epochFreelist:\n");
  _epochFreelist.dump();
  throw "Unable to allocate epoch";
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

EbEvent* EventBuilder::_event(const EbDgram* ctrb,
                              EbEvent*       after,
                              unsigned       prm)
{
  void* buffer = _eventFreelist.alloc(sizeof(EbEvent));
  if (buffer)
  {
    EbEvent* event    = ::new(buffer) EbEvent(contract(ctrb),
                                              after,
                                              ctrb,
                                              prm);
    unsigned  index  = _evIndex(ctrb->pulseId());
    _eventLut[index] = event;
    //EbEvent*& entry = _eventLut[index];
    //if (!entry)  entry = event;
    //else { printf("Event list entry %p is already allocated with key %014lx\n", entry, entry->sequence());
    //  printf("New event %p pid %014lx, index %u, mask %08lx, shift %zd\n",
    //         event, event->sequence(), index, _mask, _eventLut.size());
    //  entry->dump(0);
    //}
    return event;
  }

  fprintf(stderr, "%s:\n  Unable to allocate event: %15s %014lx\n",
          __PRETTY_FUNCTION__, TransitionId::name(ctrb->service()), ctrb->pulseId());
  //printf("  eventFreelist:\n");
  //_eventFreelist.dump();
  dump(1);
  throw "Unable to allocate event";
}

EbEvent* EventBuilder::_insert(EbEpoch*       epoch,
                               const EbDgram* ctrb,
                               EbEvent*       after,
                               unsigned       prm)
{
  const uint64_t key = ctrb->pulseId();

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
  uint64_t remaining = event->_remaining;

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
  if (entry == event)  entry = nullptr;

  delete event;
}

bool EventBuilder::_lookAhead(const EbEpoch*       epoch,
                              const EbEvent*       event,
                              const EbEvent* const due) const
{
  const EbEpoch* const lastEpoch = _pending.empty();
  const uint64_t       contract  = event->_contract;

  event = event->forward();

  while (event != due)
  {
    const EbEvent* const lastEvent = epoch->pending.empty();

    do
    {
      if (!event->_remaining && (event->_contract == contract)) // Same as matching readout groups
        return true;

      event = event->forward();

      if (event == due)
        return false;
    }
    while (event != lastEvent);

    epoch = epoch->forward();
    if (epoch == lastEpoch)  break;

    event = epoch->pending.forward();
  }

  return false;
}

void EventBuilder::_flush(const EbEvent* const due)
{
  const EbEpoch* const lastEpoch = _pending.empty();
  EbEpoch*             epoch     = _pending.forward();
  auto                 now       = fast_monotonic_clock::now();

  _tLastFlush = now;

  while (epoch != lastEpoch)
  {
    const EbEvent* const lastEvent = epoch->pending.empty();
    EbEvent*             event     = epoch->pending.forward();

    while (event != lastEvent)
    {
      // Retire all events up to a newer event, limited by due.
      // Since EbEvents are created in time order, older incomplete events can
      // be fixed up and retired when a newer complete event in the same readout
      // group (RoG) is encountered.
      if (event->_remaining)
      {
        // Time out incomplete events
        bool timedOut(std::chrono::duration_cast<std::chrono::milliseconds>(now - epoch->t0) > _eventTimeout);

        if (!timedOut)
        {
          if (!due)  return; // Not timed out; just return on timeout calls

          // If there's a newer event with the same contract/RoG and 0 remaining
          // then fix up and sweep out the current event
          // The due event has 0 remaining, so no need to lookAhead if same RoG
          if (event->_contract != due->_contract) // Same as matching RoGs
            if (!_lookAhead(epoch, event, due))  return;

          ++_fixupCnt;
        }
        else
        {
          //auto dT = std::chrono::duration_cast<std::chrono::milliseconds>(now - epoch->t0);
          //printf("Event timed out: now %ld - t0 %ld = %ld vs %d\n",
          //       now.time_since_epoch().count(), epoch->t0.time_since_epoch().count(), dT.count(), _eventTimeout.count());
          ++_tmoEvtCnt;
        }
        _missing = event->_remaining;

        //if (_verbose >= VL_EVENT)
        if (_fixupCnt + _tmoEvtCnt < 1000)
        {
          const EbDgram* dg = event->creator();
          printf("%-9s %15s %014lx, size %5zu, for remaining %016lx, RoGs %04hx, contract %016lx\n",
                 timedOut ? "Timed-out" : "Fixed-up",
                 TransitionId::name(dg->service()), event->sequence(), event->_size,
                 event->_remaining, dg->readoutGroups(), event->_contract);
        }

        _fixup(event);
      }
      if (event == due)
      {
        _retire(event);

        return;
      }

      EbEvent* next = event->forward();

      _retire(event);

      event = next;
    }

    epoch = epoch->forward();
  }
}

/*
** ++
**
**   In principle an event could sit on the pending queue forever,
**   waiting for its contract to complete. To handle this scenario,
**   the EB times-out the oldest event on the pending queue.
**   This is accomplished by periodically calling this function,
**   which attempts to flush all pending events.  Those events that
**   belong to an epoch that is suffiently old are flushed, whether
**   they are complete or not.
**
** --
*/

void EventBuilder::expired()            // Periodically called upon a timeout
{
  // Order matters: Wait one additional timeout period after _flush() has
  // emptied the EB of events before calling the application's flush().
  const EbEpoch* const lastEpoch = _pending.empty();
  EbEpoch*             epoch     = _pending.forward();

  //printf("EB::expired: epoch %p, lastEpoch %p\n", epoch, lastEpoch);
  if (epoch == lastEpoch)
  {
    flush();
    return;
  }
  else
  {
    const EbEvent* const lastEvent = epoch->pending.empty();
    EbEvent*             event     = epoch->pending.forward();
    //printf("EB::expired: event %p, lastEvent %p\n", event, lastEvent);
    if (event == lastEvent)
    {
      flush();
      return;
    }
  }

  _flush(nullptr);                      // Try to flush everything
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

void EventBuilder::process(const EbDgram* ctrb,
                           const size_t   size,
                           unsigned       prm)
{
  EbEpoch*       epoch = _match(ctrb->pulseId());
  EbEvent*       event = epoch->pending.forward();
  const EbEvent* due   = nullptr;

  while (true)
  {
    event = _insert(epoch, ctrb, event, prm);

    if (!event->_remaining)  due = event;

    if (_verbose >= VL_EVENT)
    {
      unsigned  env = ctrb->env;
      unsigned  ctl = ctrb->control();
      uint64_t  pid = ctrb->pulseId();
      uint32_t* pld = reinterpret_cast<uint32_t*>(ctrb->xtc.payload());
      size_t    sz  = sizeof(*ctrb) + ctrb->xtc.sizeofPayload();
      unsigned  src = ctrb->xtc.src.value();
      printf("EB found a ctrb                                 @ "
             "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, src %2u, pld [%08x, %08x], prm %08x, due %014lx\n",
             ctrb, ctl, pid, env, sz, src, pld[0], pld[1], prm, due ? due->sequence() : 0ul);
    }

    if (ctrb->isEOL())  break;

    ctrb = reinterpret_cast<const EbDgram*>(reinterpret_cast<const char*>(ctrb) + size);
  }

  if (due)  _flush(due);
  else
  {
    auto now    (fast_monotonic_clock::now());
    auto timeout(std::chrono::duration<int, std::milli>(100));
    if (now - _tLastFlush > timeout)  _flush(nullptr);
  }
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
