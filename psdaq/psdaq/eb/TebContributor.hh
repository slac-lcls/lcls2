#ifndef Pds_Eb_TebContributor_hh
#define Pds_Eb_TebContributor_hh

#include "eb.hh"

#include "BatchManager.hh"
#include "EbLfClient.hh"
#include "drp/spscqueue.hh"

#include <cstdint>
#include <memory>
#include <vector>
#include <atomic>
#include <thread>


namespace Pds {
  class MetricExporter;
  class EbDgram;

  namespace Eb {

    using BatchQueue = SPSCQueue<const Pds::EbDgram*>;

    class EbCtrbInBase;
    class Batch;

    class TebContributor
    {
    public:
      TebContributor(const TebCtrbParams&, const std::shared_ptr<MetricExporter>&);
      ~TebContributor();
    public:
      int         resetCounters();
      int         connect(size_t inpSizeGuess);
      int         configure();
      void        unconfigure();
      void        disconnect();
      void        startup(EbCtrbInBase&);
      void        shutdown();
      void        stop()  { _batMan.stop(); }
    public:
      void*       allocate(const Pds::TimingHeader& header, const void* appPrm);
      void        process(const Pds::EbDgram* datagram);
    public:
      void        release(uint64_t pid)        { _batMan.release(pid); }
      BatchQueue& pending()                    { return _pending; }
      const void* retrieve(uint64_t pid) const { return _batMan.retrieve(pid); }
    private:
      void       _post(const Pds::EbDgram* nonEvent) const;
      void       _post(const Pds::EbDgram* start, const Pds::EbDgram* end);
    private:
      const TebCtrbParams&      _prms;
      BatchManager              _batMan;
      EbLfClient                _transport;
      std::vector<EbLfCltLink*> _links;
      unsigned                  _id;
      unsigned                  _numEbs;
      BatchQueue                _pending; // Time ordered list of completed batches
      const Pds::EbDgram*       _batchStart;
      const Pds::EbDgram*       _batchEnd;
      bool                      _contractor;
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
