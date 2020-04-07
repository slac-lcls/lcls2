#ifndef Pds_Eb_EbAppBase_hh
#define Pds_Eb_EbAppBase_hh

#include "eb.hh"
#include "EventBuilder.hh"
#include "EbLfServer.hh"

#include "psdaq/service/MetricExporter.hh"

#include <cstdint>
#include <cstddef>
#include <string>
#include <array>
#include <vector>


namespace XtcData {
  class TimeStamp;
};

namespace Pds {
  namespace Eb {

    using u64arr_t = std::array<uint64_t, NUM_READOUT_GROUPS>;

    class EbLfSvrLink;
    class EbEvent;

    class EbAppBase : public EventBuilder
    {
    public:
      EbAppBase(const EbParams& prms,
                const uint64_t  duration,
                const unsigned  maxEntries,
                const unsigned  maxBuffers);
      virtual ~EbAppBase() {}
    public:
      int              checkEQ()  { return _transport.pollEQ(); }
    public:
      int              configure(const std::string& pfx,
                                 const EbParams&,
                                 const std::shared_ptr<MetricExporter>&);
      void             shutdown();
      int              process();
      void             trim(unsigned dst);
    public:                            // For EventBuilder
      virtual void     fixup(Pds::Eb::EbEvent* event, unsigned srcId);
      virtual uint64_t contract(const Pds::EbDgram* contrib) const;
    private:                           // Arranged in order of access frequency
      u64arr_t                  _contract;
      Pds::Eb::EbLfServer       _transport;
      std::vector<EbLfSvrLink*> _links;
      std::vector<size_t>       _bufRegSize;
      std::vector<size_t>       _maxTrSize;
      std::vector<size_t>       _maxBufSize;
      const unsigned            _maxEntries;
      const unsigned            _maxBuffers;
      //EbDummyTC                 _dummy;   // Template for TC of dummy contributions  // Revisit: ???
      const unsigned            _verbose;
      uint64_t                  _bufferCnt;
      uint64_t                  _fixupCnt;
      Pds::PromHistogram*       _fixupSrc;
      Pds::PromHistogram*       _ctrbSrc;
    private:
      void*                     _region;
      uint64_t                  _contributors;
      unsigned                  _id;
    };
  };
};

#endif

