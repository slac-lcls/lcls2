#ifndef Pds_Eb_EbAppBase_hh
#define Pds_Eb_EbAppBase_hh

#include "psdaq/eb/eb.hh"
#include "psdaq/eb/EventBuilder.hh"
#include "psdaq/service/Histogram.hh"

#include <stdint.h>
#include <cstddef>
#include <string>
#include <vector>
#include <atomic>
#include <unordered_map>

using TimePoint_t = std::chrono::steady_clock::time_point;


namespace XtcData {
  class Dgram;
  class TimeStamp;
};

namespace Pds {
  namespace Eb {

    class EbLfServer;
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
                size_t             maxInpDgSize,  // # of bytes per input datagram
                size_t             maxTrDgSize,   // # of bytes per non-event datagram
                size_t             hdrSize,       // Size of header describing the buffer
                uint64_t           contributors); // Bit list of contributors
      virtual ~EbAppBase();
    public:
      void     shutdown();
    public:
      size_t   maxInputSize() const { return _maxBufSize; }
      bool     inTrSpace(const XtcData::Dgram* dg);
      int      bufferIdx(const XtcData::Dgram* dg);
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
      const size_t                           _maxBufSize;
      void*                                  _region;
      Pds::Eb::EbLfServer*                   _transport;
      //std::vector<Pds::Eb::EbLfLink*>        _links;
      //std::unordered_map<unsigned, unsigned> _id2Idx;
      EbLfLinkMap                            _links;
      const unsigned                         _id;
      const uint64_t                         _contract;
      std::vector<size_t>                    _trOffset;
      //EbDummyTC                              _dummy;   // Template for TC of dummy contributions  // Revisit: ???
    private:
      Pds::Histogram                         _ctrbCntHist;
      Pds::Histogram                         _arrTimeHist;
      Pds::Histogram                         _pendTimeHist;
      Pds::Histogram                         _pendCallHist;
      TimePoint_t                            _pendPrevTime;
    public:
      static unsigned                        lverbose;
    };
  };
};

#endif

