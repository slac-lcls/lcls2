/**
 * @file
 * @brief EbCtrbInBase, the result-receiving half of a TEB contributor.
 */
#ifndef Pds_Eb_EbCtrbInBase_hh
#define Pds_Eb_EbCtrbInBase_hh

#include "EbLfServer.hh"
#include "eb.hh"                        // MAX_TEBS

#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <list>


namespace Pds
{
  class MetricExporter;
  class EbDgram;

  namespace Eb
  {
    class TebCtrbParams;
    class EbLfSvrLink;
    class TebContributor;
    class ResultDgram;

    /** Receives trigger result batches from the TEBs over libfabric server links, matches them with the contributor's pending input batches and calls process() for each matched event. Abstract: a subclass supplies process(). */
    class EbCtrbInBase
    {
    public:
      /** Keep a reference to the parameters and create the server transport from prms.verbose and prms.kwargs. */
      EbCtrbInBase(const TebCtrbParams&);
      /** Free the result region. */
      virtual ~EbCtrbInBase();
    public:
      /** Reset the batch, event, missing, bypass, no-progress and orphan counters; returns 0. The per-TEB state is kept. */
      int      resetCounters();
      /** Listen on prms.ifAddr:port for up to MAX_TEBS TEB connections with EbLfServer::listen(); an empty port is replaced by the bound ephemeral port. Returns 0 or the listen error. */
      int      startConnection(std::string& port);
      /** Register metrics with the exporter (if non-null) and accept one link per builder in prms.builders with linksConnect(). Returns 0 or the first error. */
      int      connect(const std::shared_ptr<MetricExporter>);
      /** Forget left-over inputs and deferred results, then for each TEB link receive the result region size, (re)allocate the shared result region if the size changed, and register it and send its description with EbLfSvrLink::setupMr(). The result buffer size is the region size divided by numBuffers. Returns 0, ENOMEM, -1 if the TEBs ask for different sizes, or the first link error. */
      int      configure(unsigned numBuffers);
      /** Does nothing; empty body. */
      void     unconfigure();
      /** Call unconfigure(), then disconnect and drop all links. */
      void     disconnect();
      /** Call disconnect(), then shut down the transport. */
      void     shutdown();
    public:
      /** Thread body: name the thread drp/TEBreceiver, call Pds::pinHotThread() (a no-op unless the process enabled thread pinning) and repeatedly wait (100 ms each) for result batches and match them with the pending inputs of ctrb, until running becomes false. Throws a C string if a TEB connection is lost (-FI_ENOTCONN) or the same negative error occurs twice in a row. */
      void     receiver(TebContributor&, std::atomic<bool>& running);
    public:
      virtual
      /** Pure virtual: handle the result of one event; index is the buffer index of the matching input datagram. Called from the receiver thread for each input matched with a result, and with a locally built persist result for inputs that bypassed the TEBs. */
      void     process(const ResultDgram& result, unsigned index) = 0;
    private:
      int     _setupMetrics(const std::shared_ptr<MetricExporter>);
      int     _linksConfigure(std::vector<EbLfSvrLink*>& links,
                              unsigned                   numBuffers,
                              const char*                name);
      int     _process(TebContributor& ctrb);
      void    _matchUp(TebContributor&    ctrb,
                       const ResultDgram* results);
      void    _defer(const ResultDgram* results);
      void    _deliverBypass(TebContributor& ctrb,
                             const EbDgram*& inputs);
      void    _deliver(TebContributor&     ctrb,
                       const ResultDgram*& results,
                       const EbDgram*&     inputs);
      void    _dump(TebContributor&    ctrb,
                    const ResultDgram* results,
                    const EbDgram*     inputs) const;
      void    _slotDump();
      bool    _orphaned(uint64_t iPid) const;
      void    _orphanDump(TebContributor& ctrb, const EbDgram* inp);
    private:
      EbLfServer                    _transport;
      std::vector<EbLfSvrLink*>     _links;
      unsigned                      _numBuffers;
      size_t                        _maxResultSize;
      const EbDgram*                _inputs;
      std::list<const ResultDgram*> _deferred;
      uint64_t                      _batchCount;
      uint64_t                      _eventCount;
      uint64_t                      _missing;
      uint64_t                      _bypassCount;
      uint64_t                      _noProgCount;
      uint64_t                      _prvNPCnt;
      // Newest batch-head pulse ID received per TEB, and a count of batches
      // from each.  A Result older than every TEB's newest can no longer be
      // in flight, the transport delivering writes to one region in order.
      uint64_t                      _newestRcvd[MAX_TEBS];
      uint64_t                      _rcvdFrom[MAX_TEBS];
      uint64_t                      _orphanCount;
      const TebCtrbParams&          _prms;
      size_t                        _regSize;
      void*                         _region;
    };
  };
};

#endif
