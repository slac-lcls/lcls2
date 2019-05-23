#ifndef Pds_Eb_TebContributor_hh
#define Pds_Eb_TebContributor_hh

#include "eb.hh"

#include "BatchManager.hh"
#include "EbLfClient.hh"

#include <cstdint>
#include <vector>
#include <atomic>
#include <thread>


namespace XtcData {
  class Dgram;
  class TimeStmp;
};

namespace Pds {
  namespace Eb {

    using BatchFifo = FifoMT<const Pds::Eb::Batch*>;

    class EbCtrbInBase;
    class Batch;
    class StatsMonitor;

    class TebContributor
    {
    public:
      TebContributor(const TebCtrbParams&, StatsMonitor&);
      ~TebContributor() {}
    public:
      int        connect(const TebCtrbParams&);
      void       startup(EbCtrbInBase&);
      void       stop();
      void       shutdown();
    public:
      void*      allocate(const XtcData::Dgram* datagram, const void* appPrm);
      void       process(const XtcData::Dgram* datagram);
      void       post(const XtcData::Dgram* nonEvent);
      void       post(const Batch* input);
    public:
      void       release(const Batch* batch) { _batMan.release(batch); }
      BatchFifo& pending()                   { return _pending; }
      Batch*     batch(unsigned idx)         { return _batMan.batch(idx); }
    private:
      const TebCtrbParams&   _prms;
      BatchManager           _batMan;
      EbLfClient             _transport;
      std::vector<EbLfLink*> _links;
      unsigned               _id;
      unsigned               _numEbs;
      BatchFifo              _pending;     // Time ordered list of completed Batches
      size_t                 _batchBase;
      uint16_t               _postFlag;
    private:
      uint64_t               _eventCount;
      uint64_t               _batchCount;
    private:
      std::atomic<bool>      _running;
      std::thread            _rcvrThread;
    };
  };
};

#endif
