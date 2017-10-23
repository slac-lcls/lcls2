#ifndef Pds_Eb_EventBuilder_hh
#define Pds_Eb_EventBuilder_hh

#include <stdint.h>

#include "psdaq/service/LinkedList.hh"
#include "psdaq/service/Timer.hh"
#include "psdaq/service/GenericPool.hh"

namespace Pds {

  class Task;
  class Datagram;

  namespace Eb {

#define EpochList LinkedList<EbEpoch>   // Notational convenience...

    class EbEpoch;
    class EbEvent;
    class EbContribution;

    class EventBuilder : public Pds::Timer
    {
    public:
      EventBuilder(unsigned epochs, unsigned entries, uint64_t mask);
      virtual ~EventBuilder();
    public:
      virtual void       fixup(EbEvent*, unsigned srcId) = 0;
      virtual void       process(EbEvent*)               = 0;
      virtual uint64_t   contract(Datagram*) const       = 0;
    protected:                          // Timer interface
      virtual void       expired();
      virtual Task*      task();
      virtual unsigned   duration()   const;
      virtual unsigned   repetitive() const;
    public:
      void               process(Datagram*);
    private:
      EbEpoch*          _match(uint64_t key);
      EbEpoch*          _epoch(uint64_t key, EbEpoch* after);
      void              _flushBefore(EbEpoch*);
      EbEpoch*          _discard(EbEpoch*);
      void              _fixup(EbEvent*);
      EbEvent*          _event(EbContribution*, EbEvent* after);
      void              _flush(EbEvent*);
      void              _retire(EbEvent*);
      EbEvent*          _insert(EbEpoch*, EbContribution*);
      EbEvent*          _insert(EbContribution*);
    private:
      EpochList         _pending;       // Listhead, Epochs with events pending
      uint64_t          _mask;          // Sequence mask
      GenericPool       _epochFreelist; // Freelist for new epochs
      GenericPool       _eventFreelist; // Freelist for new events
      Task*             _task;          // For Timer
      unsigned          _duration;      // Timer expiration rate
    };
  };
};

#endif
