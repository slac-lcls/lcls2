#ifndef Pds_Eb_EventBuilder_hh
#define Pds_Eb_EventBuilder_hh

#include <stdint.h>

#include "psdaq/service/LinkedList.hh"
#include "psdaq/service/Timer.hh"
#include "psdaq/service/GenericPool.hh"

namespace XtcData {
  class Dgram;
};

namespace Pds {

  class Task;

  namespace Eb {

#define EpochList LinkedList<EbEpoch>   // Notational convenience...

    class EbEpoch;
    class EbEvent;
    class EbContribution;

    class EventBuilder : public Pds::Timer
    {
    public:
      EventBuilder(unsigned epochs,
                   unsigned entries,
                   unsigned sources,
                   uint64_t mask);
      virtual ~EventBuilder();
    public:
      virtual void       fixup(EbEvent*, unsigned srcId)       = 0;
      virtual void       process(EbEvent*)                     = 0;
      virtual uint64_t   contract(const XtcData::Dgram*) const = 0;
    protected:                          // Timer interface
      virtual void       expired();
      virtual Task*      task();
      virtual unsigned   duration()   const;
      virtual unsigned   repetitive() const;
    public:
      void               process(const XtcData::Dgram*, uint64_t appParam);
    public:
      void               dump(unsigned detail);
    private:
      EbEpoch*          _match(uint64_t key);
      EbEpoch*          _epoch(uint64_t key, EbEpoch* after);
      void              _flushBefore(EbEpoch*);
      EbEpoch*          _discard(EbEpoch*);
      void              _fixup(EbEvent*);
      EbEvent*          _event(const XtcData::Dgram*, uint64_t param, EbEvent* after);
      void              _flush(EbEvent*);
      void              _retire(EbEvent*);
      EbEvent*          _insert(EbEpoch*, const XtcData::Dgram*, uint64_t param);
      EbEvent*          _insert(const XtcData::Dgram*, uint64_t param);
    private:
      friend class EbEvent;
    private:
      EpochList         _pending;       // Listhead, Epochs with events pending
      uint64_t          _mask;          // Sequence mask
      GenericPool       _epochFreelist; // Freelist for new epochs
      GenericPool       _eventFreelist; // Freelist for new events
      GenericPool       _cntrbFreelist; // Freelist for new contributions
      Task*             _task;          // For Timer
      unsigned          _duration;      // Timer expiration rate
    };
  };
};

#endif
