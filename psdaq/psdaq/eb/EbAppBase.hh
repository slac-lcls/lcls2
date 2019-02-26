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

using TimePoint_t = std::chrono::steady_clock::time_point;


namespace XtcData {
  class Dgram;
  class TimeStamp;
};

namespace Pds {
  namespace Eb {

    class EbLfLink;
    class EbEvent;

    class EbAppBase : public EventBuilder
    {
    public:
      EbAppBase(const EbParams& prms);
    public:
      const uint64_t&  rxPending() const { return _transport.pending(); }
    public:
      int              connect(const EbParams&);
      int              process();
      void             shutdown();
    public:                          // For EventBuilder
      virtual void     fixup(Pds::Eb::EbEvent* event, unsigned srcId);
      virtual uint64_t contract(const XtcData::Dgram* contrib) const;
    private:
      void            _updateHists(TimePoint_t               t0,
                                   TimePoint_t               t1,
                                   const XtcData::TimeStamp& stamp);
    private:                           // Arranged in order of access frequency
      uint64_t                 _defContract;
      std::array<uint64_t, 16> _contract;
      Pds::Eb::EbLfServer      _transport;
      std::vector<EbLfLink*>   _links;
      size_t                   _trSize;
      size_t                   _maxTrSize;
      std::vector<size_t>      _maxBufSize;
      //EbDummyTC                _dummy;   // Template for TC of dummy contributions  // Revisit: ???
      unsigned                 _verbose;
    private:
      Pds::Histogram           _ctrbCntHist;
      Pds::Histogram           _arrTimeHist;
      Pds::Histogram           _pendTimeHist;
      Pds::Histogram           _pendCallHist;
      TimePoint_t              _pendPrevTime;
    private:
      void*                    _region;
      unsigned                 _id;
    };
  };
};

#endif

