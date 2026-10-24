/**
 * @file
 * @brief MebContributor, which copies datagrams into local buffers and RDMA-writes them to monitor event builders (MEBs).
 */
#ifndef Pds_Eb_MebContributor_hh
#define Pds_Eb_MebContributor_hh

#include "eb.hh"
#include "EbLfClient.hh"

#include <cstdint>
#include <memory>
#include <vector>
#include <list>


namespace Pds {
  class EbDgram;
  class MetricExporter;

  namespace Eb {

    class EbLfCltLink;

    /** Sends L1Accepts to one chosen MEB buffer and transitions to every MEB, over libfabric client links. Each link gets a local region of event buffers followed by MEB_TR_BUFFERS transition buffers. */
    class MebContributor
    {
    public:
      /** Keep a reference to the parameters; the event buffer size is prms.maxEvSize rounded up to whole pages. */
      MebContributor(const MebCtrbParams&);
    public:
      /** Reset the event and transition counters; returns 0. */
      int  resetCounters();
      /** Register metrics with the exporter (if non-null), size the per-link vectors from prms.addrs and connect to all MEBs with linksConnect(). Returns 0 or the first error. */
      int  connect(const std::shared_ptr<MetricExporter>);
      /** For each link, (re)allocate the local region if its size changed (maxEvents of that MEB times the event buffer size, plus the transition buffers), exchange region information with EbLfCltLink::prepare() and refill the list of MEB_TR_BUFFERS transition buffer indices. Sets enabled() true and returns 0, or returns the first error (ENOMEM if allocation fails). */
      int  configure();
      /** Set enabled() to false. */
      void unconfigure();
      /** Call unconfigure(), disconnect and drop all links, and set the ID to -1. */
      void disconnect();
      /** Call disconnect() (harmless if already done). */
      void shutdown();
      /** Return true between a successful configure() and unconfigure(). */
      bool enabled() const { return _enabled; }
    public:
      /** Send a transition to every MEB: for each link take a free transition buffer index (polling the link, waiting up to 5 s if none is cached), copy the datagram into that local buffer and RDMA-write it. Aborts if the datagram is larger than maxTrSize, its source is not this contributor's ID, or no buffer index arrives; a failed write is only logged. Returns 0. */
      int  post(const Pds::EbDgram* dataDatagram); // Transitions
      /** Send an L1Accept to the MEB and buffer index encoded (ImmData src and idx) in destination: copy it into the matching local buffer and RDMA-write it. Aborts on an invalid MEB or index, an oversize datagram (checked after the copy), a source ID mismatch, an out-of-range buffer, a non-advancing pulse ID, or a failed write. Returns 0. */
      int  post(const Pds::EbDgram* dataDatagram,
                uint32_t            destination);  // L1Accepts
    private:
      int _setupMetrics(const std::shared_ptr<MetricExporter>);
      int _linksConfigure(const MebCtrbParams&       prms,
                          std::vector<EbLfCltLink*>& links,
                          const char*                peer);
    public:
      /** List of free transition buffer indices (one list per MEB link). */
      using listU32_t = std::list<uint32_t>;
    private:
      const MebCtrbParams&      _prms;
      size_t                    _maxEvSize;
      size_t                    _maxTrSize;
      EbLfClient                _transport;
      std::vector<EbLfCltLink*> _links;
      std::vector<void*>        _region;
      std::vector<size_t>       _regSize;
      std::vector<size_t>       _bufRegSize;
      std::vector<listU32_t>    _trBuffers;
      unsigned                  _id;
      bool                      _enabled;
      unsigned                  _verbose;
      uint64_t                  _previousPid;
    private:
      uint64_t                  _eventCount;
      uint64_t                  _trCount;
    };
  };
};

#endif
