#ifndef Pds_Eb_TebContributor_hh
#define Pds_Eb_TebContributor_hh

#include "eb.hh"

#include "BatchManager.hh"
#include "EbLfClient.hh"

#include <cstdint>
#include <vector>
#include <chrono>
#include <atomic>
#include <thread>


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

    class EbCtrbInBase;
    class Batch;

    class TebContributor : public BatchManager
    {
    public:
      TebContributor(const TebCtrbParams&);
      virtual ~TebContributor() {}
    public:
      int      connect(const TebCtrbParams&);
      void     startup(EbCtrbInBase&);
      void     stop()  { _running = false; }
      void     shutdown();
    public:
      bool     process(const XtcData::Dgram* datagram, const void* appPrm);
      void     post(const XtcData::Dgram* nonEvent);
    public:                             // For BatchManager
      virtual void post(const Batch* input);
    public:
      const uint64_t& batchCount()   const { return _batchCount;  }
      const uint64_t& txPending()    const { return _transport.pending(); }
      unsigned        inFlightCnt()  const { return _inFlightOcc; }
    private:
      void    _receiver(EbCtrbInBase&);
    protected:
      const TebCtrbParams&   _prms;
    private:
      EbLfClient             _transport;
      std::vector<EbLfLink*> _links;
      unsigned               _id;
      unsigned               _numEbs;
      size_t                 _batchBase;
    private:
      uint64_t               _batchCount;
      std::atomic<unsigned>  _inFlightOcc;
    private:
      std::atomic<bool>      _running;
      std::thread            _rcvrThread;
    };
  };
};

#endif
