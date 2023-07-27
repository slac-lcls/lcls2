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

  class EbDgram;

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
                const unsigned msTimeout);
      virtual ~EbAppBase();
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
    protected:
      const std::vector<size_t> bufferSizes() const;
    public:                            // For EventBuilder
      virtual void     fixup(Pds::Eb::EbEvent* event, unsigned srcId);
      virtual uint64_t contract(const Pds::EbDgram* contrib) const;
    private:
      int              _linksConfigure(const EbParams&            prms,
                                       std::vector<EbLfSvrLink*>& links,
                                       const char*                name);
    private:                           // Arranged in order of access frequency
      u64arr_t                  _contract;
      Pds::Eb::EbLfServer       _transport;
      std::vector<EbLfSvrLink*> _links;
      std::vector<size_t>       _bufRegSize;
      std::vector<size_t>       _maxTrSize;
      std::vector<size_t>       _maxBufSize;
      unsigned                  _maxEntries;
      unsigned                  _maxEvBuffers;
      unsigned                  _maxTrBuffers;
      unsigned&                 _verbose;
      uint64_t                  _ebTime;
      uint64_t                  _bufferCnt;
      PromHisto_t               _fixupSrc;
      PromHisto_t               _ctrbSrc;
    private:
      std::vector<size_t>       _regSize;
      std::vector<void*>        _region;
      uint64_t                  _contributors;
      uint64_t                  _idxSrcs;
      unsigned                  _id;
      MetricExporter_t          _exporter;
      const std::string         _pfx;
    };
  };
};


inline
const std::vector<size_t> Pds::Eb::EbAppBase::bufferSizes() const
{
  std::vector<size_t> bufSizes(_bufRegSize.size());

  for (unsigned i = 0; i < _bufRegSize.size(); ++i)
    bufSizes[i] = _bufRegSize[i] / _maxEvBuffers / _maxEntries;

  return bufSizes;
}

#endif
