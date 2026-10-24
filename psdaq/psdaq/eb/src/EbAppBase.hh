/**
 * @file
 * @brief EbAppBase, the common base of the trigger and monitor event builders: an EventBuilder fed by libfabric server links from the contributors.
 */
#ifndef Pds_Eb_EbAppBase_hh
#define Pds_Eb_EbAppBase_hh

#include "eb.hh"
#include "EventBuilder.hh"
#include "EbLfServer.hh"

#include "psdaq/service/MetricExporter.hh"
#include "psdaq/service/Collection.hh"

#include <cstdint>
#include <cstddef>
#include <string>
#include <array>
#include <vector>


/** Namespace of the xtcdata library (XTC datagram types); psdaq headers only forward-declare parts of it. */
namespace XtcData {
  class TimeStamp;
};

namespace Pds {

  class EbDgram;

  namespace Eb {

    class EbLfSvrLink;
    class EbEvent;

    /** EventBuilder fed by inbound libfabric server links from the contributors (DRPs). Each contributor gets a local region holding its input buffers followed by its transition buffers. */
    class EbAppBase : public EventBuilder
    {
    public:
      /** Array of NUM_READOUT_GROUPS uint64_t values (contributor bit lists per readout group). */
      using u64arr_t         = std::array<uint64_t, NUM_READOUT_GROUPS>;
      /** Shared pointer to a Pds::PromHistogram. */
      using PromHisto_t      = std::shared_ptr<Pds::PromHistogram>;
      /** Shared pointer to a Pds::MetricExporter. */
      using MetricExporter_t = std::shared_ptr<Pds::MetricExporter>;

    public:
      /** Construct the EventBuilder with the timeout from the eb_timeout entry of prms.kwargs (inserted as EB_TMO_MS if absent), create the server transport and connect a ZMQ PUSH socket to tcp://collSrv:(CollectionApp::zmq_base_port + partition) for asynchronous error messages. pfx labels metrics and messages. */
      EbAppBase(EbParams& prms, const std::string& pfx);
      /** Free the per-contributor regions. */
      virtual ~EbAppBase();
    public:
      /** Reset the buffer counter and the notification pulse ID, clear the histograms if present and reset the EventBuilder counters; returns 0. */
      int              resetCounters();
      /** Listen for nLinks contributor connections on ifAddr:port with EbLfServer::listen(); if port is empty, the bound ephemeral port is written back into port. Returns 0 or the listen error. */
      int              startConnection(const std::string& ifAddr,
                                       std::string&       port,
                                       unsigned           nLinks);
      /** Size the per-contributor vectors from the number of bits in prms.contributors, initialize the EventBuilder for (EB_TMO_MS / 1000) * (maxEvBuffers / maxEntries) + maxTrBuffers epochs, register metrics if exporter is non-null and accept the contributor links with linksConnect(). Returns 0 or the first error. */
      int              connect(unsigned maxEvBuffers, unsigned maxTrBuffers, const MetricExporter_t);
      /** For each link, receive the buffer size requested by the contributor, (re)allocate the local region if its size changed (that size times prms.numBuffers for the contributor, plus maxTrBuffers transition buffers of prms.maxTrSize), then register it and send its description with EbLfSvrLink::setupMr(). Returns 0 or the first error (ENOMEM if allocation fails). */
      int              configure();
      /** Call EventBuilder::clear(). */
      void             unconfigure();
      /** Call unconfigure(), disconnect all links and reset the ID, the contract table and the buffer size vectors. */
      void             disconnect();
      /** Call disconnect(), then shut down the transport. */
      void             shutdown();
      /** Wait up to 100 ms for one buffer notification and pass the referenced batch or transition datagram, chosen by the immediate data flags, to EventBuilder::process() (the immediate data is passed only for contributors in prms.indexSources). On -FI_EAGAIN, call EventBuilder::expired() to time out incomplete events. Returns 0 or the negative pend() error; aborts on an invalid source ID or a failed sanity check (source, index, region bounds, size, pulse ID order). */
      int              process();
      /** For each datagram in [begin, end), send its transition buffer index back to the contributor named by its source (immediate data NoResponse_Transition with this builder's ID). Aborts on an invalid contributor ID or index; a failed post is only logged. */
      void             post(const EbDgram* const* begin,
                            const EbDgram** const end);
      /** Remove contributor dst from the contract of every readout group. */
      void             trim(unsigned dst);
    protected:
      const std::vector<size_t>& bufferSizes() const;
    public:                            // For EventBuilder
      /** Handle event missing contributor srcId: set DroppedContribution damage, send asynchronous error messages naming the missing contributors of the first such event to the collection server, log the first 50 fixups and timeouts, and record srcId in the fixup histogram if present. */
      virtual void     fixup(Pds::Eb::EbEvent* event, unsigned srcId);
      /** Return the OR of the contracts (contributor bit lists) of all readout groups set in the readoutGroups() of contrib. */
      virtual uint64_t contract(const Pds::EbDgram* contrib) const;
    private:
      int              _setupMetrics(const MetricExporter_t);
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
      std::vector<uint64_t>     _lastPid;
      uint64_t                  _bufferCnt;
      PromHisto_t               _fixupSrc;
      PromHisto_t               _ctrbSrc;
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
