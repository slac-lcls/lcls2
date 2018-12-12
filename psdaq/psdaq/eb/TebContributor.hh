#ifndef Pds_Eb_TebContributor_hh
#define Pds_Eb_TebContributor_hh

#include "eb.hh"

#include "BatchManager.hh"
#include "EbLfClient.hh"

#include "psdaq/service/Histogram.hh"

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>
#include <atomic>


namespace std {
  class thread;
};

namespace XtcData {
  class Dgram;
  class TimeStmp;
};

namespace Pds {
  namespace Eb {

    // Put these in our namespace so as not to break others
    using TimePoint_t = std::chrono::steady_clock::time_point;
    using Duration_t  = std::chrono::steady_clock::duration;
    using us_t        = std::chrono::microseconds;
    using ns_t        = std::chrono::nanoseconds;

    class EbLfLink;
    class EbCtrbInBase;
    class Batch;
    class StatsMonitor;

    using EbLfLinkMap = std::unordered_map<unsigned, Pds::Eb::EbLfLink*>;

    class TebContributor : public BatchManager
    {
    public:
      TebContributor(const TebCtrbParams&);
      virtual ~TebContributor();
    public:
      void     startup(EbCtrbInBase&);
      void     shutdown();
    public:
      bool     process(const XtcData::Dgram* datagram, const void* appPrm);
      void     post(const XtcData::Dgram* nonEvent);
    public:                             // For BatchManager
      virtual void post(const Batch* input);
    public:
      const uint64_t& batchCount()   const { return _batchCount;  }
      const uint64_t& txPending()    const { return _transport->pending(); }
      unsigned        inFlightCnt()  const { return _inFlightOcc; }
    private:
      void    _receiver(EbCtrbInBase&);
      void    _updateHists(TimePoint_t               t0,
                           TimePoint_t               t1,
                           const XtcData::TimeStamp& stamp);
    private:
      EbLfClient*            _transport;
      EbLfLinkMap            _links;
      unsigned*              _idx2Id;
      const unsigned         _id;
      const unsigned         _numEbs;
    private:
      uint64_t               _batchCount;
    private:
      std::atomic<unsigned>  _inFlightOcc;
      Histogram              _inFlightHist;
      Histogram              _depTimeHist;
      Histogram              _postTimeHist;
      Histogram              _postCallHist;
      TimePoint_t            _postPrevTime;
    private:
      std::atomic<bool>      _running;
      std::thread*           _rcvrThread;
    protected:
      const TebCtrbParams&   _prms;
    };
  };
};

#endif
