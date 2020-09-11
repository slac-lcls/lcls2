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

    class EbLfSvrLink;
    class EbEvent;

    class EbAppBase : public EventBuilder
    {
    public:
      using u64arr_t         = std::array<uint64_t, NUM_READOUT_GROUPS>;
      using PromHisto_t      = std::shared_ptr<Pds::PromHistogram>;
      using MetricExporter_t = std::shared_ptr<Pds::MetricExporter>;

    public:
      EbAppBase(const EbParams& prms,
                const MetricExporter_t&,
                const std::string& pfx,
                const uint64_t duration,
                const unsigned maxEntries,
                const unsigned maxBuffers);
      virtual ~EbAppBase();
    public:
      int              checkEQ()  { return _transport.pollEQ(); }
    public:
      int              resetCounters();
      int              startConnection(const std::string& ifAddr,
                                       std::string&       port,
                                       unsigned           nLinks);
      int              connect(const EbParams& prms, size_t inpSizeGuess);
      int              configure(const EbParams& prms);
      void             unconfigure();
      void             disconnect();
      void             shutdown();
      int              process();
      void             post(const EbDgram* const* begin,
                            const EbDgram** const end);
      void             trim(unsigned dst);
    public:                            // For EventBuilder
      virtual void     fixup(Pds::Eb::EbEvent* event, unsigned srcId);
      virtual uint64_t contract(const Pds::EbDgram* contrib) const;
    private:
      int              _linksConfigure(const EbParams&            prms,
                                       std::vector<EbLfSvrLink*>& links,
                                       unsigned                   id,
                                       const char*                name);
    private:                           // Arranged in order of access frequency
      u64arr_t                  _contract;
      Pds::Eb::EbLfServer       _transport;
      std::vector<EbLfSvrLink*> _links;
      std::vector<size_t>       _bufRegSize;
      std::vector<size_t>       _maxTrSize;
      std::vector<size_t>       _maxBufSize;
      const unsigned            _maxEntries;
      const unsigned            _maxBuffers;
      const unsigned&           _verbose;
      uint64_t                  _bufferCnt;
      PromHisto_t               _fixupSrc;
      PromHisto_t               _ctrbSrc;
    private:
      std::vector<size_t>       _regSize;
      std::vector<void*>        _region;
      uint64_t                  _contributors;
      unsigned                  _id;
      const MetricExporter_t&   _exporter;
      const std::string&        _pfx;
    };
  };
};

#endif

