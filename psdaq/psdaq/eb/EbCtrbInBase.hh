#ifndef Pds_Eb_EbCtrbInBase_hh
#define Pds_Eb_EbCtrbInBase_hh

#include "EbLfServer.hh"

#include <memory>
#include <vector>
#include <atomic>


class MetricExporter;

namespace Pds
{
  namespace Eb
  {
    class TebCtrbParams;
    class EbLfSvrLink;
    class Batch;
    class TebContributor;
    class ResultDgram;

    class EbCtrbInBase
    {
    public:
      EbCtrbInBase(const TebCtrbParams&, const std::shared_ptr<MetricExporter>&);
      virtual ~EbCtrbInBase() {}
    public:
      int      configure(const TebCtrbParams&);
    public:
      void     receiver(TebContributor&, std::atomic<bool>& running);
    public:
      virtual void process(const ResultDgram& result, const void* input) = 0;
    private:
      void    _initialize(const char* who);
      void    _shutdown();
      int     _process(TebContributor& ctrb);
      void    _pairUp(TebContributor&    ctrb,
                      unsigned           idx,
                      const ResultDgram* result);
      void    _deliver(TebContributor&    ctrb,
                       const ResultDgram* results,
                       const Batch*       inputs);
    private:
      EbLfServer                _transport;
      std::vector<EbLfSvrLink*> _links;
      size_t                    _maxBatchSize;
      uint64_t                  _batchCount;
      uint64_t                  _eventCount;
      uint64_t                  _deliverCount;
      const TebCtrbParams&      _prms;
      void*                     _region;
    };
  };
};

#endif
