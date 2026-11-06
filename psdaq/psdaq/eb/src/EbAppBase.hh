#ifndef Pds_Eb_EbAppBase_hh
#define Pds_Eb_EbAppBase_hh

#include "eb.hh"
#include "EventBuilder.hh"
#include "EbLfServer.hh"
#include "SlotTracker.hh"

#include "psdaq/service/MetricExporter.hh"
#include "psdaq/service/Collection.hh"

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
      EbAppBase(EbParams& prms, const std::string& pfx);
      virtual ~EbAppBase();
    public:
      int              resetCounters();
      int              startConnection(const std::string& ifAddr,
                                       std::string&       port,
                                       unsigned           nLinks);
      int              connect(unsigned maxEvBuffers, unsigned maxTrBuffers, const MetricExporter_t);
      int              configure();
      void             unconfigure();
      void             disconnect();
      void             shutdown();
      int              process();
      void             post(const EbDgram* const* begin,
                            const EbDgram** const end);
      void             trim(unsigned dst);
    protected:
      const std::vector<size_t>& bufferSizes() const;
    public:                            // For EventBuilder
      virtual void     fixup(Pds::Eb::EbEvent* event, unsigned srcId);
      virtual uint64_t contract(const Pds::EbDgram* contrib) const;
      virtual void     retiring(const Pds::Eb::EbEvent* event);
    protected:
      // Lets an application refuse a contribution, e.g., to an index it never handed out
      virtual bool     accepting(unsigned src, unsigned idx, const Pds::EbDgram*) { return true; }
      void             slotDump()  const { _slots.dump(_pfx.c_str()); }
      bool             slotCheck() const { return _slotCheck; }
    private:
      int              _setupMetrics(const MetricExporter_t);
      int              _linksConfigure(const EbParams&            prms,
                                       std::vector<EbLfSvrLink*>& links,
                                       const char*                name);
      size_t           _slot(unsigned src, const Pds::EbDgram*) const;
      void             _claim(unsigned src, const Pds::EbDgram* idg, const void* end, uint32_t imm);
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
      std::vector<uint64_t>     _lastPid;
      uint64_t                  _bufferCnt;
      PromHisto_t               _fixupSrc;
      PromHisto_t               _ctrbSrc;
      SlotTracker               _slots;
      bool                      _slotCheck;
    private:
      std::vector<size_t>       _regSize;
      std::vector<void*>        _region;
      uint64_t                  _idxSrcs;
      unsigned                  _id;
      MetricExporter_t          _exporter;
      ZmqContext                _context;
      ZmqSocket                 _notifySocket;
      uint64_t                  _notifyPid;
      const std::string         _pfx;
      const EbParams&           _prms;
    };
  };
};


inline
const std::vector<size_t>& Pds::Eb::EbAppBase::bufferSizes() const
{
  return _maxBufSize;
}

#endif
