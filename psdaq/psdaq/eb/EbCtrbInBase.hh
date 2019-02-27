#ifndef Pds_Eb_EbCtrbInBase_hh
#define Pds_Eb_EbCtrbInBase_hh

#include "psdaq/eb/EbLfServer.hh"

#include <chrono>
#include <vector>


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

    class EbCtrbInBase
    {
    public:
      EbCtrbInBase(const TebCtrbParams&);
      virtual ~EbCtrbInBase() {}
    public:
      const uint64_t& rxPending() const { return _transport.pending(); }
    public:
      int      connect(const TebCtrbParams&);
      int      process(BatchManager& batMan);
      void     shutdown();
    public:
      size_t   maxBatchSize() const { return _maxBatchSize; }
    public:
      virtual void process(const XtcData::Dgram* result, const void* input) = 0;
    private:
      void    _initialize(const char* who);
    private:
      EbLfServer             _transport;
      std::vector<EbLfLink*> _links;
      size_t                 _maxBatchSize;
    protected:
      const TebCtrbParams&   _prms;
    private:
      void*                  _region;
    };
  };
};

#endif
