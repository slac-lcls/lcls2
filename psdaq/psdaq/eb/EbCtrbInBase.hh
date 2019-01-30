#ifndef Pds_Eb_EbCtrbInBase_hh
#define Pds_Eb_EbCtrbInBase_hh

#include "psdaq/service/Histogram.hh"
#include "psdaq/eb/EbLfServer.hh"

#include <chrono>
#include <vector>
#include <unordered_map>
#include <string>


namespace XtcData
{
  class Dgram;
  class TimeStamp;
};

namespace Pds
{
  namespace Eb
  {
    using TimePoint_t = std::chrono::steady_clock::time_point;

    class TebCtrbParams;
    class EbLfLink;
    class Batch;
    class BatchManager;

    using UmapEbLfLink = std::unordered_map<unsigned, Pds::Eb::EbLfLink*>;

    class EbCtrbInBase
    {
    public:
      EbCtrbInBase(const TebCtrbParams&);
      virtual ~EbCtrbInBase();
    public:
      const uint64_t& rxPending() const { return _transport->pending(); }
    public:
      void     shutdown();
      int      process(BatchManager& batMan);
    public:
      size_t   maxBatchSize() const { return _maxBatchSize; }
    public:
      virtual void process(const XtcData::Dgram* result, const void* input) = 0;
    private:
      void    _initialize(const char* who);
      void    _updateHists(TimePoint_t               t0,
                           TimePoint_t               t1,
                           const XtcData::TimeStamp& stamp);
    protected:
      const TebCtrbParams& _prms;
    private:
      const unsigned       _numEbs;
      size_t               _maxBatchSize;
      std::vector<void*>   _regions;
      EbLfServer*          _transport;
      UmapEbLfLink         _links;
    private:
      Histogram            _ebCntHist;
      Histogram            _rttHist;
      Histogram            _pendTimeHist;
      Histogram            _pendCallHist;
      TimePoint_t          _pendPrevTime;
    };
  };
};

#endif
