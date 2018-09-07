#ifndef Pds_Eb_EbCtrbInBase_hh
#define Pds_Eb_EbCtrbInBase_hh

#include "psdaq/service/Histogram.hh"

#include <chrono>
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

    class EbCtrbParams;
    class EbLfLink;
    class EbLfServer;
    class Batch;
    class BatchManager;

    class EbCtrbInBase
    {
    public:
      EbCtrbInBase(const EbCtrbParams&);
      virtual ~EbCtrbInBase();
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
    private:
      const unsigned         _numEbs;
      const size_t           _maxBatchSize;
      void*                  _region;
      EbLfServer*            _transport;
      //std::vector<EbLfLink*> _links;
      std::unordered_map<unsigned,
                         Pds::Eb::EbLfLink*> _links;
    private:
      Histogram              _ebCntHist;
      Histogram              _rttHist;
      Histogram              _pendTimeHist;
      Histogram              _pendCallHist;
      TimePoint_t            _pendPrevTime;
    protected:
      const EbCtrbParams&    _prms;
    };
  };
};

#endif
