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

    using EbLfLinkMap = std::unordered_map<unsigned, Pds::Eb::EbLfLink*>;

    class EbAppBase : public EventBuilder
    {
    public:
      EbAppBase(const char*        ifAddr,        // NIC address
                const std::string& port,          // Port number
                unsigned           id,            // Unique instance ID in range 0 - 63
                uint64_t           duration,      // Buffer accumulation duration
                unsigned           maxBuffers,    // # of concurrent buffers
                unsigned           maxEntries,    // # of datagrams per buffer
                size_t             maxInDgSize,   // # of bytes per input datagram
                size_t             maxTrDgSize,   // # of bytes per non-event datagram
                uint64_t           contributors); // Bit list of contributors
      virtual ~EbAppBase();
    public:
      const uint64_t& rxPending() const { return _transport->pending(); }
    public:
      void     shutdown();
    public:
      size_t   maxInputSize() const { return _maxBufSize; }
    public:
      void     process();
    public:                          // For EventBuilder
      virtual void     fixup(Pds::Eb::EbEvent* event, unsigned srcId);
      virtual uint64_t contract(const XtcData::Dgram* contrib) const;
    private:
      void    _updateHists(TimePoint_t               t0,
                           TimePoint_t               t1,
                           const XtcData::TimeStamp& stamp);
    private:
      const size_t             _maxBufSize;
      void*                    _region;
      Pds::Eb::EbLfServer*     _transport;
      EbLfLinkMap              _links;
      const unsigned           _id;
      const uint64_t           _defContract;
      std::array<uint64_t, 16> _contract;
      std::vector<size_t>      _trOffset;
      //EbDummyTC                _dummy;   // Template for TC of dummy contributions  // Revisit: ???
    private:
      Pds::Histogram           _ctrbCntHist;
      Pds::Histogram           _arrTimeHist;
      Pds::Histogram           _pendTimeHist;
      Pds::Histogram           _pendCallHist;
      TimePoint_t              _pendPrevTime;
    public:
      static unsigned          lverbose;
    };
  };
};

#endif

