#ifndef Pds_Eb_EventBuilder_hh
#define Pds_Eb_EventBuilder_hh

#include <stdint.h>
#include <vector>

#include "psdaq/service/LinkedList.hh"
#include "psdaq/service/GenericPool.hh"

namespace XtcData {
  class Dgram;
};

namespace Pds {

  namespace Eb {

    class EbEpoch;
    class EbEvent;

    class EventBuilder
    {
    public:
      EventBuilder(unsigned epochs,
                   unsigned entries,
                   unsigned sources,
                   uint64_t mask,
                   unsigned verbose);
      virtual ~EventBuilder();
    public:
      virtual void       fixup(EbEvent*, unsigned srcId)      = 0;
      virtual void       process(EbEvent*)                    = 0;
      virtual uint64_t   contracts(const XtcData::Dgram*,
                                   uint64_t& receivers) const = 0;
    public:
      void               expired();
    public:
      void               process(const XtcData::Dgram*, unsigned prm);
    public:
      void               clear();
      void               dump(unsigned detail) const;
      const uint64_t&    epochAllocCnt() const;
      const uint64_t&    epochFreeCnt()  const;
      const uint64_t&    eventAllocCnt() const;
      const uint64_t&    eventFreeCnt()  const;
    private:
      unsigned          _epIndex(uint64_t key) const;
      unsigned          _evIndex(uint64_t key) const;
    private:
      EbEpoch*          _match(uint64_t key);
      EbEpoch*          _epoch(uint64_t key, EbEpoch* after);
      void              _flushBefore(EbEpoch*);
      EbEpoch*          _discard(EbEpoch*);
      void              _fixup(EbEvent*);
      EbEvent*          _event(const XtcData::Dgram*, EbEvent* after, unsigned prm);
      void              _flush(EbEvent*);
      void              _retire(EbEvent*);
      EbEvent*          _insert(EbEpoch*, const XtcData::Dgram*, EbEvent*, unsigned prm);
    private:
      friend class EbEvent;
    private:
      LinkedList<EbEpoch>   _pending;       // Listhead, Epochs with events pending
      const uint64_t        _mask;          // Sequence mask
      GenericPool           _epochFreelist; // Freelist for new epochs
      std::vector<EbEpoch*> _epochLut;      // LUT of allocated epochs
      GenericPool           _eventFreelist; // Freelist for new events
      std::vector<EbEvent*> _eventLut;      // LUT of allocated events
      unsigned              _verbose;       // Print progress info
    };
  };
};

inline const uint64_t& Pds::Eb::EventBuilder::epochAllocCnt() const
{
  return _epochFreelist.numberofAllocs();
}

inline const uint64_t& Pds::Eb::EventBuilder::epochFreeCnt() const
{
  return _epochFreelist.numberofFrees();
}

inline const uint64_t& Pds::Eb::EventBuilder::eventAllocCnt() const
{
  return _eventFreelist.numberofAllocs();
}

inline const uint64_t& Pds::Eb::EventBuilder::eventFreeCnt() const
{
  return _eventFreelist.numberofFrees();
}

#endif
