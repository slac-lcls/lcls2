#ifndef Pds_Eb_EbCtrbInBase_hh
#define Pds_Eb_EbCtrbInBase_hh

#include "EbLfServer.hh"

#include <memory>
#include <vector>
#include <atomic>


class MetricExporter;

namespace XtcData
{
  class Dgram;
  class TimeStamp;
};

namespace Pds
{
  namespace Eb
  {
    class TebCtrbParams;
    class EbLfLink;
    class Batch;
    class TebContributor;

    class EbCtrbInBase
    {
    public:
      EbCtrbInBase(const TebCtrbParams&, std::shared_ptr<MetricExporter>&);
      virtual ~EbCtrbInBase() {}
    public:
      int      connect(const TebCtrbParams&);
      int      process(TebContributor& ctrb);
      void     shutdown();
    public:
      size_t   maxBatchSize() const { return _maxBatchSize; }
      void     receiver(TebContributor&, std::atomic<bool>& running);
    public:
      virtual void process(const XtcData::Dgram* result, const void* input) = 0;
    private:
      void    _initialize(const char* who);
      void    _pairUp(TebContributor&       ctrb,
                      unsigned              idx,
                      const XtcData::Dgram* result);
      void    _process(TebContributor&       ctrb,
                       const XtcData::Dgram* results,
                       const Batch*          inputs);
    private:
      EbLfServer             _transport;
      std::vector<EbLfLink*> _links;
      size_t                 _maxBatchSize;
      uint64_t               _batchCount;
      uint64_t               _eventCount;
      const TebCtrbParams&   _prms;
      void*                  _region;
    };
  };
};

#endif
