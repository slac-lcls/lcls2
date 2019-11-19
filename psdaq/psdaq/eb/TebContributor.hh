#ifndef Pds_Eb_TebContributor_hh
#define Pds_Eb_TebContributor_hh

#include "eb.hh"

#include "BatchManager.hh"
#include "EbLfClient.hh"
#include "psdaq/service/Fifo.hh"

#include <cstdint>
#include <memory>
#include <vector>
#include <atomic>
#include <thread>


class MetricExporter;

namespace XtcData {
  class EbDgram;
  class TimeStmp;
};

namespace Pds {
  namespace Eb {

    using BatchFifo = FifoMT<const Pds::Eb::Batch*>;

    class EbCtrbInBase;
    class Batch;

    class TebContributor
    {
    public:
      TebContributor(const TebCtrbParams&, const std::shared_ptr<MetricExporter>&);
      ~TebContributor() {}
    public:
      int        configure(const TebCtrbParams&);
      void       startup(EbCtrbInBase&);
      void       shutdown();
    public:
      void*      allocate(const XtcData::EbDgram* header, const void* appPrm);
      void       process(const XtcData::EbDgram* datagram);
    public:
      void       release(const Batch* batch) { _batMan.release(batch); }
      BatchFifo& pending()                   { return _pending; }
      Batch*     batch(unsigned idx)         { return _batMan.batch(idx); }
    private:
      void       _post(const XtcData::EbDgram* nonEvent) const;
      void       _post(const Batch* input) const;
    private:
      const TebCtrbParams&      _prms;
      BatchManager              _batMan;
      EbLfClient                _transport;
      std::vector<EbLfCltLink*> _links;
      unsigned                  _id;
      unsigned                  _numEbs;
      BatchFifo                 _pending; // Time ordered list of completed Batches
      size_t                    _batchBase;
      Batch*                    _batch;
    private:
      mutable uint64_t          _eventCount;
      mutable uint64_t          _batchCount;
    private:
      std::atomic<bool>         _running;
      std::thread               _rcvrThread;
    };
  };
};

#endif
