#ifndef Pds_Eb_TebContributor_hh
#define Pds_Eb_TebContributor_hh

#include "eb.hh"

#include "psdaq/service/EbDgram.hh"

#include "BatchManager.hh"
#include "EbLfClient.hh"
#include "drp/spscqueue.hh"

#include <cstdint>
#include <memory>
#include <vector>
#include <atomic>
#include <thread>
#include <list>


namespace Pds {
  class MetricExporter;
  class EbDgram;

  namespace Eb {

    using BatchQueue = SPSCQueue<const Pds::EbDgram*>;

    class EbCtrbInBase;
    struct Batch
    {
      Batch(const Pds::EbDgram* dgram, bool contractor_) :
        start(dgram), end(dgram), contractor(contractor_) {}

      const Pds::EbDgram* start;
      const Pds::EbDgram* end;
      bool                contractor;
    };

    class TebContributor
    {
    public:
      TebContributor(const TebCtrbParams&,
                     unsigned numBuffers,
                     const std::shared_ptr<MetricExporter>&);
      ~TebContributor();
    public:
      int         resetCounters();
      int         connect(size_t inpSizeGuess);
      int         configure();
      void        unconfigure();
      void        disconnect();
      void        startup(EbCtrbInBase&);
      void        shutdown();
    public:
      void        process(const Pds::EbDgram* datagram);
      unsigned    index(const Pds::EbDgram* datagram) const;
      void*       fetch(unsigned index);
      void        process(unsigned index);
    public:
      BatchQueue& pending()  { return _pending; }
    private:
      void       _post(const Pds::EbDgram* nonEvent);
      void       _post(const Batch& batch);
    public:
      using listU32_t = std::list<uint32_t>;
    private:
      const TebCtrbParams&      _prms;
      BatchManager              _batMan;
      EbLfClient                _transport;
      std::vector<EbLfCltLink*> _links;
      std::vector<listU32_t >   _trBuffers;
      size_t                    _trRegOffset;
      unsigned                  _id;
      unsigned                  _numEbs;
      BatchQueue                _pending; // Time ordered list of completed batches
      Batch                     _batch;
      uint64_t                  _previousPid;
    private:
      mutable uint64_t          _eventCount;
      mutable uint64_t          _batchCount;
      mutable uint64_t          _pendingSize;
      mutable int64_t           _latency;
    private:
      std::atomic<bool>         _running;
      std::thread               _rcvrThread;
    };
  };
};

inline
unsigned Pds::Eb::TebContributor::index(const Pds::EbDgram* dgram) const
{
  unsigned offset = reinterpret_cast<const char*>(dgram) -
                    static_cast<const char*>(_batMan.batchRegion());
  uint32_t idx    = offset / _prms.maxInputSize;
  return idx;
}

inline
void* Pds::Eb::TebContributor::fetch(unsigned index)
{
  return _batMan.fetch(index);
}

inline
void Pds::Eb::TebContributor::process(unsigned index)
{
  process(static_cast<Pds::EbDgram*>(fetch(index)));
}

#endif
