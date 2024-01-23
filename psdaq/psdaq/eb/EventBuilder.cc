#include "eb.hh"
#include "EventBuilder.hh"
#include "EbEpoch.hh"
#include "EbEvent.hh"

#include "psdaq/service/Task.hh"
#include "xtcdata/xtc/Dgram.hh"

#include <stdlib.h>
#include <new>
#include <chrono>

#define UNLIKELY(expr)  __builtin_expect(!!(expr), 0)
#define LIKELY(expr)    __builtin_expect(!!(expr), 1)

using namespace XtcData;
using namespace Pds;
using namespace Pds::Eb;

using ms_t = std::chrono::milliseconds;
using us_t = std::chrono::microseconds;

static constexpr unsigned CLS = 64;     // Cache Line Size


EventBuilder::EventBuilder(unsigned        timeout,
                           const unsigned& verbose) :
  _pending     (),
  _eventTimeout(uint64_t(timeout) * 1000000ul), // Convert to ns
  _tmoEvtCnt   (0),
  _fixupCnt    (0),
  _missing     (0),
  _epochOccCnt (0),
  _eventOccCnt (0),
  _age         (0),
  _ebTime      (0),
  _verbose     (verbose)
{
}

EventBuilder::~EventBuilder()
{
}

int EventBuilder::initialize(unsigned epochs,
                             unsigned entries,
                             unsigned sources,
                             uint64_t duration)
{
  if (duration & (duration - 1))
  {
    fprintf(stderr, "%s:\n  Epoch duration (%016lx) must be a power of 2\n",
            __func__, duration);
    return 1;
  }
  _mask = ~PulseId(duration - 1).pulseId();
  _maxEntries = entries;

  // Revisit the factor of 2: it seems like the sum of the values over all RoGs
  // may be closer to the right answer
  auto nep = 2 * epochs;
  auto nev = 2 * epochs * entries;

  // In the following, the freelist sizes are double what may seem like the right
  // value so that the skew in DRP contribution arrivals creating new events can
  // be accomodated, especially during "pause/resume" tests, etc.
  auto epSize = sizeof(EbEpoch) + entries * sizeof(EbEvent*);
  auto evSize = sizeof(EbEvent) + sources * sizeof(EbDgram*);
  _epochFreelist = std::make_unique<GenericPool>(epSize, nep, CLS);
  _eventFreelist = std::make_unique<GenericPool>(evSize, nev, CLS);

  _epochLut.resize(nep, nullptr);
  _eventLut.resize(nev, nullptr);

  printf("*** EB Epoch list size %zu\n", _epochLut.size());
  printf("*** EB Event list size %zu\n", _eventLut.size());

  _arrTime.resize(sources);

  return 0;
}

void EventBuilder::resetCounters()
{
  if (_eventFreelist)  _eventFreelist->clearCounters();
  if (_epochFreelist)  _epochFreelist->clearCounters();
  _tmoEvtCnt   = 0;
  _fixupCnt    = 0;
  _missing     = 0;
  _epochOccCnt = 0;
  _eventOccCnt = 0;
  _age         = 0;
  _ebTime      = 0;

  for (auto& arrTime : _arrTime)  arrTime = 0;
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

      const uint64_t key   = event->sequence();
      unsigned       index = _evIndex(key);
      _eventLut[index] = nullptr;

      delete event;

      event = next;
    }
    epoch = epoch->forward();
  }

  _flushBefore(_pending.empty());

  resetCounters();

  std::fill(_epochLut.begin(), _epochLut.end(), nullptr);
  std::fill(_eventLut.begin(), _eventLut.end(), nullptr);
}

inline
unsigned EventBuilder::_epIndex(uint64_t key) const
{
  return (key >> __builtin_ctzl(_mask)) % _epochLut.size();
}

inline
unsigned EventBuilder::_evIndex(uint64_t key) const
{
  return key % _eventLut.size();
}

EbEpoch* EventBuilder::_discard(EbEpoch* epoch)
{
  EbEpoch* next = epoch->reverse();

  const uint64_t key   = epoch->key;
  unsigned       index = _epIndex(key);
  EbEpoch*&      entry = _epochLut[index];
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
  void* buffer = _epochFreelist->alloc(sizeof(EbEpoch));
  if (LIKELY(buffer))
  {
    EbEpoch*  epoch = ::new(buffer) EbEpoch(key, after);

    unsigned  index = _epIndex(key);
    EbEpoch*& entry = _epochLut[index];
    if (!entry)  entry = epoch;

    return epoch;
  }

  fprintf(stderr, "%s:\n  Unable to allocate epoch: key %014lx\n",
          __PRETTY_FUNCTION__, key);
  printf(" epochFreelist:\n");
  _epochFreelist->dump();
  dump(1);
  while(1);                             // Hang so we can inspect
  abort();
}

EbEpoch* EventBuilder::_match(uint64_t inKey)
{
  const uint64_t key = inKey & _mask;

  unsigned index = _epIndex(key);
  EbEpoch* entry = _epochLut[index];
  if (entry && (entry->key == key))  return entry;

  const EbEpoch* const empty = _pending.empty();
  EbEpoch*             epoch = _pending.reverse();

  while (epoch != empty)
  {
    uint64_t epochKey = epoch->key;

    if (epochKey == key)
    {
      if (!entry)  entry = epoch;
      return epoch;
    }
    if (epochKey <  key) break;
    epoch = epoch->reverse();
  }

  _flushBefore(epoch);
  return _epoch(key, epoch);
}

EbEvent* EventBuilder::_event(EbEpoch*            epoch,
                              const EbDgram*      ctrb,
                              EbEvent*            after,
                              unsigned            imm,
                              const time_point_t& t0)
{
  void* buffer = _eventFreelist->alloc(sizeof(EbEvent));
  if (LIKELY(buffer))
  {
    EbEvent* event = ::new(buffer) EbEvent(contract(ctrb),
                                           after,
                                           ctrb,
                                           imm,
                                           t0);

    const uint64_t key   = ctrb->pulseId();
    unsigned       index = _evIndex(key);
    _eventLut[index] = event;

    return event;
  }

  fprintf(stderr, "%s:\n  Unable to allocate event: %15s %014lx\n",
          __PRETTY_FUNCTION__, TransitionId::name(ctrb->service()), ctrb->pulseId());
  printf("  eventFreelist:\n");
  _eventFreelist->dump();
  dump(1);
  while(1);                             // Hang so we can inspect
  abort();
}

EbEvent* EventBuilder::_insert(EbEpoch*            epoch,
                               const EbDgram*      ctrb,
                               EbEvent*            after,
                               unsigned            imm,
                               const time_point_t& t0)
{
  const uint64_t key = ctrb->pulseId();

  unsigned index = _evIndex(key);
  EbEvent* event = _eventLut[index];
  if (event && (event->sequence() == key))  return event->_add(ctrb, imm);

  bool                 reversed = false;
  const EbEvent* const empty    = epoch->pending.empty();
  event                         = after;

  while (event != empty)
  {
    const uint64_t eventKey = event->sequence();

    if (key == eventKey) return event->_add(ctrb, imm);
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

  return _event(epoch, ctrb, after, imm, t0);
}

void EventBuilder::_fixup(EbEvent*             event,
                          ns_t                 age,
                          const EbEvent* const due)
{
  uint64_t remaining = event->_remaining;
  _missing = remaining;

  // remaining != 0 whenever _fixup() is called
  do
  {
    unsigned srcId = __builtin_ffsl(remaining) - 1;

    fixup(event, srcId);

    remaining &= ~(1ull << srcId);
  }
  while (remaining);

  if (age < _eventTimeout)  ++_fixupCnt;
  else                      ++_tmoEvtCnt;

  //if (_verbose >= VL_EVENT)
  if (_fixupCnt + _tmoEvtCnt < 100)
  {
    const EbDgram* dg = event->creator();
    printf("%-10s %15s %014lx, size %5zu, for  remaining %016lx, RoGs %04hx, contract %016lx, age %ld ms, tmo %ld ms\n",
           age < _eventTimeout ? "Fixed-up" : "Timed-out",
           TransitionId::name(dg->service()), event->sequence(), event->_size,
           event->_remaining, dg->readoutGroups(), event->_contract,
           std::chrono::duration_cast<ms_t>(age).count(),
           std::chrono::duration_cast<ms_t>(_eventTimeout).count());
    if (age < _eventTimeout)
      printf("Flushed by %15s %014lx, size %5zu, with remaining %016lx, RoGs %04hx, contract %016lx\n",
             TransitionId::name(due->creator()->service()), due->sequence(),
             due->_size, due->_remaining, dg->readoutGroups(), due->_contract);
  }
}

void EventBuilder::_retire(EbEpoch* epoch, EbEvent* event)
{
  event->disconnect();

  process(event);

  auto age{fast_monotonic_clock::now(CLOCK_MONOTONIC) - event->_t0};
  _age = std::chrono::duration_cast<ns_t>(age).count();

  const uint64_t key   = event->sequence();
  unsigned       index = _evIndex(key);
  EbEvent*&      entry = _eventLut[index];
  if (entry == event)  entry = nullptr;

  delete event;
}

void EventBuilder::_flush(const EbEvent* const due)
{
  const EbEpoch* const lastEpoch = _pending.empty();
  EbEpoch*             epoch     = _pending.forward();
  auto                 now       = fast_monotonic_clock::now(CLOCK_MONOTONIC);

  _tLastFlush = now;

  while (epoch != lastEpoch)
  {
    const EbEvent* const lastEvent = epoch->pending.empty();
    EbEvent*             event     = epoch->pending.forward();

    while (event != lastEvent)
    {
      const auto age{now - event->_t0};

      // Retire all events up to a newer event, limited by due.
      // Since EbEvents are created in time order, older incomplete events can
      // be fixed up and retired when a newer complete event in the same readout
      // group (RoG) is encountered.
      if (event->_remaining)
      {
        // The due event may be incomplete if progress is stalled in which case
        // events in the same RoG need to be be timed out
        if ((event->_contract != due->_contract) || due->_remaining)
        {
          // Time out incomplete events
          if (age < _eventTimeout)  return;
        }

        _fixup(event, age, due);
      }
      if (event == due)
      {
        _retire(epoch, event);

        return;
      }

      EbEvent* next = event->forward();

      _retire(epoch, event);

      event = next;
    }

    epoch = epoch->forward();
  }
}

void EventBuilder::_flush()
{
  const EbEpoch* const lastEpoch = _pending.empty();
  EbEpoch*             epoch     = _pending.reverse();

  if (epoch != lastEpoch)
  {
    const EbEvent* const lastEvent = epoch->pending.empty();
    EbEvent*             event     = epoch->pending.reverse();

    if (event != lastEvent)  _flush(event); // Most recent event in the system
  }
}

// This is called to time out incomplete events when contributions are
// still flowing but never become due (i.e., events are always incomplete).
void EventBuilder::_tryFlush()
{
  const ms_t tmo{100};
  auto       now{fast_monotonic_clock::now(CLOCK_MONOTONIC)};
  if (now - _tLastFlush > tmo)  _flush();
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

  if (epoch == lastEpoch)
  {
    flush();
    return;
  }
  else
  {
    const EbEvent* const lastEvent = epoch->pending.empty();
    EbEvent*             event     = epoch->pending.forward();

    if (event == lastEvent)
    {
      flush();
      return;
    }
  }

  _flush();                             // Try to flush everything
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

void EventBuilder::process(const EbDgram*    buffer,
                           const size_t      size,
                           unsigned          immData,
                           const void* const end)
{
  auto t0{fast_monotonic_clock::now(CLOCK_MONOTONIC)};

  const EbDgram* ctrb  = buffer;
  unsigned       imm   = immData;
  EbEpoch*       epoch = _match(ctrb->pulseId());
  EbEvent*       event = epoch->pending.forward();
  const EbEvent* due   = nullptr;

  for (unsigned i = 0; i < _maxEntries; ++i)
  {
    event = _insert(epoch, ctrb, event, imm, t0);

    if (!event->_remaining)
    {
      if (due && (event->_contract != due->_contract))  _flush(due);
      due = event;
    }

    uint64_t  pid = ctrb->pulseId();
    unsigned  env = ctrb->env;
    unsigned  src = ctrb->xtc.src.value();
    if (UNLIKELY(_verbose >= VL_EVENT))
    {
      unsigned  ctl = ctrb->control();
      uint32_t* pld = reinterpret_cast<uint32_t*>(ctrb->xtc.payload());
      size_t    sz  = sizeof(*ctrb) + ctrb->xtc.sizeofPayload();
      printf("EB found a ctrb                                 @ "
             "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, src %2u, pld [%08x, %08x], imm %08x, due %014lx%s\n",
             ctrb, ctl, pid, env, sz, src, pld[0], pld[1], imm, due ? due->sequence() : 0ul, ctrb->isEOL() ? ", EOL" : "");
    }

    if (ctrb->isEOL())  break;

    ctrb = reinterpret_cast<const EbDgram*>(reinterpret_cast<const char*>(ctrb) + size);
    imm++;
    if ((ctrb > end) || (i == _maxEntries - 1))
    {
      fprintf(stderr, "%s:\n  Error: EOL not seen before buffer end, last pid %014lx, env %08x, src %u\n"
              "  buffer %p, end %p, ctrb %p, entry size, %zu, immData %08x, imm %08x, i %u\n",
              __PRETTY_FUNCTION__, pid, env, src, buffer, end, ctrb, size, immData, imm, i);
    }
  }

  if (due)  _flush(due);     // Attempt to flush everything up to the due event
  else      _tryFlush();     // Periodically flush when no events are completing

  auto t1{fast_monotonic_clock::now(CLOCK_MONOTONIC)};
  auto src = ctrb->xtc.src.value();     // Same for all ctrbs in a batch
  _arrTime[src] = std::chrono::duration_cast<ns_t>(t0 - event->_t0).count();
  _ebTime       = std::chrono::duration_cast<ns_t>(t1 - t0).count();
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
  _epochFreelist->dump();

  printf("Event Builder event pool:\n");
  _eventFreelist->dump();
}
