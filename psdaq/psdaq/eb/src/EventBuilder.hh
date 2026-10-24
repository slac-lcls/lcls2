/**
 * @file
 * @brief EventBuilder, the abstract core that collects contributions with equal pulse IDs into events and delivers them in order.
 */
#ifndef Pds_Eb_EventBuilder_hh
#define Pds_Eb_EventBuilder_hh

#include <stdint.h>
#include <vector>

#include "psdaq/service/LinkedList.hh"
#include "psdaq/service/GenericPool.hh"
#include "psdaq/service/fast_monotonic_clock.hh"

namespace XtcData {
  class TimeStamp;
};
namespace Pds {
  class EbDgram;
};

namespace Pds {

  namespace Eb {

    class EbEpoch;
    class EbEvent;

    /** Abstract event builder. Contributions with the same pulse ID are collected into an EbEvent (events are grouped into epochs of a fixed duration), and events are passed to process(EbEvent*) in pulse ID order once their contract is met or they are forced out. Subclasses supply fixup(), process(EbEvent*) and contract(). */
    class EventBuilder
    {
    public:
      /** Set the event timeout to timeout milliseconds (0 means effectively no timeout) and keep verbose by reference; prints the timeout to stderr. The protected initialize() must be called before contributions are processed. */
      EventBuilder(unsigned        timeout,
                   const unsigned& verbose);
      /** Does nothing; empty body. */
      virtual ~EventBuilder();
    protected:
      int                initialize(unsigned epochs,
                                    unsigned entries,
                                    unsigned sources,
                                    uint64_t duration);
    public:
      /** Hook called by expired() when nothing is pending; the default does nothing. */
      virtual void       flush() {}
      /** Pure virtual: called once for each missing contributor srcId of an event that is being forced out (timed out or flushed by a newer complete event). */
      virtual void       fixup(EbEvent*, unsigned srcId)     = 0;
      /** Pure virtual: handle a completed or fixed-up event; called in pulse ID order just before the event is freed. */
      virtual void       process(EbEvent*)                   = 0;
      /** Pure virtual: return the bit list of contributors expected for the event that this contribution starts. */
      virtual uint64_t   contract(const Pds::EbDgram*) const = 0;
    public:
      /** Periodic timeout handler. If no event is pending, call flush(); otherwise flush pending events oldest first, fixing up incomplete events older than the timeout and stopping at the first incomplete event that may still complete. */
      void               expired();
    public:
      /** Insert a batch of contributions starting at dgrams (entries bufSize bytes apart, ending at the EOL entry or after the maximum number of entries) into their events, with immediate data imm incremented per entry; end is only used to report a missing EOL. Completed events, and older events subject to the timeout rules, are then delivered; if none completed, a flush is tried at most every 100 ms. Also updates the arrival-time and processing-time metrics. */
      void               process(const Pds::EbDgram* dgrams,
                                 const size_t        bufSize,
                                 unsigned            imm,
                                 const void* const   end);
    public:
      /** Zero the pool counters, the timeout and fixup counts, the missing bit list, the occupancy counts, the event age, the processing time and the arrival times. */
      void               resetCounters();
      /** Delete all pending events, discard the now-empty epochs, reset the counters and clear the epoch and event lookup tables. */
      void               clear();
      /** Print the pending epochs (and their events, depending on detail) and the state of both object pools to stderr. */
      void               dump(unsigned detail) const;
      /** Return the number of allocations from the epoch pool (0 before initialize()). */
      const uint64_t     epochAllocCnt()  const;
      /** Return the number of frees to the epoch pool (0 before initialize()). */
      const uint64_t     epochFreeCnt()   const;
      /** Return the number of epochs in use (allocations minus frees), also caching it. */
      const int64_t      epochOccCnt()    const;
      /** Return the number of allocations from the event pool (0 before initialize()). */
      const uint64_t     eventAllocCnt()  const;
      /** Return the number of frees to the event pool (0 before initialize()). */
      const uint64_t     eventFreeCnt()   const;
      /** Return the number of events in use (allocations minus frees), also caching it. */
      const int64_t      eventOccCnt()    const;
      /** Return the number of objects in the event pool (0 before initialize()); returned by value. */
      const uint64_t     eventPoolDepth() const; // Right: not a ref
      /** Return the number of incomplete events forced out because they were older than the timeout. */
      const uint64_t     timeoutCnt()     const;
      /** Return the number of incomplete events forced out before reaching the timeout. */
      const uint64_t     fixupCnt()       const;
      /** Return the bit list of contributors still missing from the most recently fixed-up event. */
      const uint64_t     missing()        const;
      /** Return the most recently recorded event age in nanoseconds (updated while flushing and when an event is retired). */
      const int64_t      eventAge()       const;
      /** Return the time spent in the most recent process(dgrams, ...) call, in nanoseconds. */
      const int64_t      ebTime()         const;
      /** Return, for contributor src, the time in nanoseconds from the creation of the last event in its most recent batch to the start of processing that batch; 0 if src is out of range. */
      const int64_t      arrTime(unsigned src) const;
    private:
      friend class EbEvent;
    public:
      /** Time point of fast_monotonic_clock. */
      using time_point_t = std::chrono::time_point<fast_monotonic_clock>;
      /** Alias for std::chrono::nanoseconds. */
      using ns_t         = std::chrono::nanoseconds;
    private:
      unsigned          _epIndex(uint64_t key) const;
      unsigned          _evIndex(uint64_t key) const;
    private:
      EbEpoch*          _match(uint64_t key);
      EbEpoch*          _epoch(uint64_t key, EbEpoch* after);
      void              _flushBefore(EbEpoch*);
      EbEpoch*          _discard(EbEpoch*);
      EbEvent*          _event(EbEpoch*,
                               const Pds::EbDgram*,
                               EbEvent* after,
                               unsigned imm,
                               const time_point_t&);
      EbEvent*          _insert(EbEpoch*,
                                const Pds::EbDgram*,
                                EbEvent*,
                                unsigned imm,
                                const time_point_t&);
      void              _fixup(EbEvent*, ns_t age, const EbEvent* const due);
      void              _retire(EbEvent*);
      void              _flush(const EbEvent* const due);
      void              _flush();
      void              _tryFlush();
    private:
      LinkedList<EbEpoch>          _pending;       // Listhead, Epochs with events pending
      time_point_t                 _tLastFlush;    // Starting time of timeout
      uint64_t                     _mask;          // Sequence mask
      unsigned                     _maxEntries;    // Maximum number of entries per buffer/batch
      std::unique_ptr<GenericPool> _epochFreelist; // Freelist for new epochs
      std::vector<EbEpoch*>        _epochLut;      // LUT of allocated epochs
      std::unique_ptr<GenericPool> _eventFreelist; // Freelist for new events
      std::vector<EbEvent*>        _eventLut;      // LUT of allocated events
      const ns_t                   _eventTimeout;  // Maximum event age in ms
      mutable uint64_t             _tmoEvtCnt;     // Count of timed out events
      mutable uint64_t             _fixupCnt;      // Count of flushed   events
      mutable uint64_t             _missing;       // Bit list of missing contributors
      mutable int64_t              _epochOccCnt;   // Number of epochs in use
      mutable int64_t              _eventOccCnt;   // Number of events in use
      mutable int64_t              _age;           // Event age
      mutable int64_t              _ebTime;        // Processing time
      std::vector<int64_t>         _arrTime;       // Contribution arrival time
      const unsigned&              _verbose;       // Print progress info
    };
  };
};

inline const uint64_t Pds::Eb::EventBuilder::epochAllocCnt() const
{
  return _epochFreelist ? _epochFreelist->numberofAllocs() : 0;
}

inline const uint64_t Pds::Eb::EventBuilder::epochFreeCnt() const
{
  return _epochFreelist ? _epochFreelist->numberofFrees() : 0;
}

// Revisit: This one is not terribly interesting and mirrors eventOccCnt()
inline const int64_t Pds::Eb::EventBuilder::epochOccCnt() const
{
  _epochOccCnt = epochAllocCnt() - epochFreeCnt();

  return _epochOccCnt;
}

inline const uint64_t Pds::Eb::EventBuilder::eventAllocCnt() const
{
  return _eventFreelist ? _eventFreelist->numberofAllocs() : 0;
}

inline const uint64_t Pds::Eb::EventBuilder::eventFreeCnt() const
{
  return _eventFreelist ? _eventFreelist->numberofFrees() : 0;
}

inline const int64_t Pds::Eb::EventBuilder::eventOccCnt() const
{
  _eventOccCnt = eventAllocCnt() - eventFreeCnt();

  return _eventOccCnt;
}

inline const uint64_t Pds::Eb::EventBuilder::eventPoolDepth() const
{
  // Return a copy of the value instead of a reference
  // since it is nominally called only once by MetricExporter
  return _eventFreelist ? _eventFreelist->numberofObjects() : 0;
}

inline const uint64_t Pds::Eb::EventBuilder::timeoutCnt() const
{
  return _tmoEvtCnt;
}

inline const uint64_t Pds::Eb::EventBuilder::fixupCnt() const
{
  return _fixupCnt;
}

inline const uint64_t Pds::Eb::EventBuilder::missing() const
{
  return _missing;
}

inline const int64_t Pds::Eb::EventBuilder::eventAge() const
{
  return _age;
}

inline const int64_t Pds::Eb::EventBuilder::ebTime() const
{
  return _ebTime;
}

inline const int64_t Pds::Eb::EventBuilder::arrTime(unsigned src) const
{
  return (src < _arrTime.size()) ? _arrTime[src] : 0;
}

#endif
