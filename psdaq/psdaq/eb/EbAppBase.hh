#ifndef Pds_Eb_EbAppBase_hh
#define Pds_Eb_EbAppBase_hh

#include "eb.hh"
#include "EventBuilder.hh"
#include "EbLfServer.hh"
#include "psdaq/service/Histogram.hh"

#include <stdint.h>
#include <cstddef>
#include <string>
#include <array>
#include <vector>
#include <unordered_map>

using TimePoint_t = std::chrono::steady_clock::time_point;


namespace XtcData {
  class Dgram;
  class TimeStamp;
};

namespace Pds {
  namespace Eb {

    class EbLfLink;
    class EbEvent;

    using UmapEbLfLink = std::unordered_map<unsigned, Pds::Eb::EbLfLink*>;
    using UmapSize_t   = std::unordered_map<unsigned, size_t>;

    class EbAppBase : public EventBuilder
    {
    public:
      EbAppBase(const EbParams& prms);
      virtual ~EbAppBase();
    public:
      const uint64_t& rxPending() const { return _transport->pending(); }
    public:
      void     shutdown();
    public:
      int      process();
    public:                          // For EventBuilder
      virtual void     fixup(Pds::Eb::EbEvent* event, unsigned srcId);
      virtual uint64_t contract(const XtcData::Dgram* contrib) const;
    private:
      void    _updateHists(TimePoint_t               t0,
                           TimePoint_t               t1,
                           const XtcData::TimeStamp& stamp);
    private:
      std::vector<void*>       _regions;
      UmapSize_t               _maxBufSize;
      std::vector<size_t>      _trOffset;
      size_t                   _trSize;
      Pds::Eb::EbLfServer*     _transport;
      UmapEbLfLink             _links;
      const unsigned           _id;
      const uint64_t           _defContract;
      std::array<uint64_t, 16> _contract;
      //EbDummyTC                _dummy;   // Template for TC of dummy contributions  // Revisit: ???
      const unsigned           _verbose;
    private:
      Pds::Histogram           _ctrbCntHist;
      Pds::Histogram           _arrTimeHist;
      Pds::Histogram           _pendTimeHist;
      Pds::Histogram           _pendCallHist;
      TimePoint_t              _pendPrevTime;
    };
  };
};

#endif

