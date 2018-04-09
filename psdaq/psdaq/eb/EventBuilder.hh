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

    class EbEpoch;
    class EbEvent;

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
      void               process    (const XtcData::Dgram*);
      void               processBulk(const XtcData::Dgram*);
    public:
      void               dump(unsigned detail);
    private:
      EbEpoch*          _match(uint64_t key);
      EbEpoch*          _epoch(uint64_t key, EbEpoch* after);
      void              _flushBefore(EbEpoch*);
      EbEpoch*          _discard(EbEpoch*);
      void              _fixup(EbEvent*);
      EbEvent*          _event(const XtcData::Dgram*, EbEvent* after);
      void              _flush(EbEvent*);
      void              _retire(EbEvent*);
      EbEvent*          _insert(EbEpoch*, const XtcData::Dgram*);
      EbEvent*          _insert(EbEpoch*, const XtcData::Dgram*, EbEvent*);
    private:
      friend class EbEvent;
    private:
      LinkedList<EbEpoch> _pending;       // Listhead, Epochs with events pending
      uint64_t            _mask;          // Sequence mask
      GenericPool         _epochFreelist; // Freelist for new epochs
      GenericPool         _eventFreelist; // Freelist for new events
      Task*               _timerTask;     // For Timer
      unsigned            _duration;      // Timer expiration rate
    };
  };
};

#endif
