/**
 * @file
 * @brief EbLfServer, which accepts libfabric client links (EbLfSvrLink) and receives their RDMA-write notifications, and helpers that start and connect a set of such links.
 */
#ifndef Pds_Eb_EbLfServer_hh
#define Pds_Eb_EbLfServer_hh

#include "EbLfLink.hh"

#include <stdint.h>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>

struct fi_cq_data_entry;

namespace Pds {

  namespace Fabrics {
    class PassiveEndpoint;
    class CompletionQueue;
  };

  namespace Eb {

    /** Map from a raw libfabric endpoint to its server link. */
    using LinkMap = std::unordered_map<fid_ep*, EbLfSvrLink*>;

    /** Accepts client connections on a passive endpoint and receives their notifications. All links share one event queue and one receive completion queue. */
    class EbLfServer
    {
    public:
      /** Construct with default libfabric hints; nothing is opened until listen(). */
      EbLfServer(const unsigned& verbose);
      /** Construct with libfabric hints built from kwargs (see Fabrics::Info); nothing is opened until listen(). */
      EbLfServer(const unsigned&                           verbose,
                 const std::map<std::string, std::string>& kwargs);
      /** Delete all links, the receive completion queue, the event queue and the passive endpoint. */
      ~EbLfServer();
    public:
      /** Create a passive endpoint on addr:port and start listening with a backlog of nLinks. If port is empty, the OS chooses a port and it is written back into port. Returns 0 or an error number. */
      int  listen(const std::string& addr,    // Interface to use
                  std::string&       port,    // Port being listened on
                  unsigned           nLinks); // Max number of links
      /** Accept one connection, waiting up to msTmo ms, and store a new EbLfSvrLink in the first argument. The shared event queue and receive completion queue (nLinks times the provider receive size) are created on first use. Returns 0 or an error number. */
      int  connect(EbLfSvrLink**, unsigned nLinks, int msTmo = -1);
      /** Delete the link and close its endpoint; the shared queues are deleted when no links remain. Does nothing for a null link; always returns FI_SUCCESS. */
      int  disconnect(EbLfSvrLink*);
      /** Delete the passive endpoint. */
      void shutdown();
      /** Get one receive completion into the entry. The server polls first; after 1 ms without a completion it returns -FI_EAGAIN and switches to blocking waits of msTmo ms, going back to polling after a completion. Returns the positive completion count, -FI_EAGAIN, -FI_ENOTCONN if a client disconnected, or another negative error. */
      int  pend(fi_cq_data_entry*, int msTmo);
      /** Call pend(fi_cq_data_entry*, msTmo) and store the completion's op_context in context; it is stored even when no completion was read. */
      int  pend(void** context, int msTmo);
      /** Call pend(fi_cq_data_entry*, msTmo) and store the completion's immediate data in data; it is stored even when no completion was read. */
      int  pend(uint64_t* data, int msTmo);
      /** Read one receive completion in the current polling or waiting mode (see pend()) and store its immediate data in data (0 if none). Returns the positive count, -FI_EAGAIN, -FI_ENOTCONN if a client disconnected, or another negative error. */
      int  poll(uint64_t* data);
      /** Read one event from the shared event queue without blocking. Returns -FI_ENOTCONN for an FI_SHUTDOWN from a known link (or if there is no queue), -FI_ENOKEY for one from an unknown endpoint, the queue error for other events or read errors, and FI_SUCCESS when no event is available. */
      int  pollEQ();
      /** Register (or reuse) a memory region for size bytes at region on the passive endpoint's fabric; returns -1 if there is no passive endpoint. */
      int  setupMr(void* region, size_t size);
    public:
      /** Return the shared pending word (incremented while pend() or a link's timed poll() is waiting). */
      uint64_t pending() const { return _pending; }
      /** Return the shared posting word, a bit list of peer IDs currently posting. */
      uint64_t posting() const { return _posting; }
    private:
      int _poll(fi_cq_data_entry*, uint64_t flags);
    private:                              // Arranged in order of access frequency
      Fabrics::EventQueue*      _eq;      // Event Queue
      Fabrics::CompletionQueue* _rxcq;    // Receive Completion Queue
      int                       _tmo;     // Timeout for polling or waiting
      const unsigned&           _verbose; // Print some stuff if set
    private:
      volatile uint64_t         _pending; // Flag set when currently pending
      volatile uint64_t         _posting; // Bit list of IDs currently posting
      uint64_t                  _noCtxCnt;// Rx completions naming no link
    private:
      Fabrics::PassiveEndpoint* _pep;     // EP for establishing connections
      Fabrics::MemoryRegion*    _mr;      // Keep track of the MR
      LinkMap                   _linkByEp;// Map to retrieve link given raw EP
      Fabrics::Info             _info;    // Connection options
    };

    // --- Revisit: The following maybe better belongs somewhere else

    /** Call transport.listen(ifAddr, port, nLinks) and log an error naming name on failure; returns 0 or the error. */
    int linksStart(EbLfServer&        transport,
                   const std::string& ifAddr,
                   std::string&       port,
                   unsigned           nLinks,
                   const char*        name);
    /** Accept links.size() connections (14750 ms timeout each), then exchange IDs on each and store each link at the index of its remote ID. name labels the peers in messages. Returns 0 or the first error. */
    int linksConnect(EbLfServer&                transport,
                     std::vector<EbLfSvrLink*>& links,
                     unsigned                   id,
                     const char*                name);
  };
};

inline
int Pds::Eb::EbLfServer::_poll(fi_cq_data_entry* cqEntry, uint64_t flags)
{
  ssize_t rc;

  if (!_rxcq)  return -FI_ENOTCONN; // Not connected (see connect())

  // Polling favors latency, waiting favors throughput
  if (!_tmo)
  {
    rc = _rxcq->comp(cqEntry, 1);   // Uses much less kernel time than comp_wait() with tmo = 0
  }
  else
  {
    rc = _rxcq->comp_wait(cqEntry, 1, _tmo);
    if (rc > 0)  _tmo = 0;          // Switch to polling after successful completion
  }

  if (rc > 0)
  {
    if (cqEntry->op_context)
    {
      auto link = static_cast<Pds::Eb::EbLfLink*>(cqEntry->op_context);
      link->postCompRecv(rc);
    }
    else
    {
      // The normal case for Results: a remote write (FI_RMA|FI_REMOTE_WRITE|
      // FI_REMOTE_CQ_DATA, no FI_RECV) consumes no posted receive buffer, so
      // there is no credit to replenish and no context to name a link by.
      // Counted only because the count tells the two cases apart.
      ++_noCtxCnt;
    }

#ifdef DBG
    if ((cqEntry->flags & flags) != flags)
    {
      fprintf(stderr, "%s:\n  Expected   CQ entry:\n"
                      "  count %zd, got flags %016lx vs %016lx, data = %08lx\n"
                      "  ctx   %p, len %zd, buf %p\n",
              __PRETTY_FUNCTION__, rc, cqEntry->flags, flags, cqEntry->data,
              cqEntry->op_context, cqEntry->len, cqEntry->buf);
    }
#endif
  }

  return rc;
}

inline
int Pds::Eb::EbLfServer::pend(void** ctx, int msTmo)
{
  fi_cq_data_entry cqEntry;

  int rc = pend(&cqEntry, msTmo);
  *ctx = cqEntry.op_context;

  return rc;
}

inline
int Pds::Eb::EbLfServer::pend(uint64_t* data, int msTmo)
{
  fi_cq_data_entry cqEntry;

  int rc = pend(&cqEntry, msTmo);
  *data = cqEntry.data;

  return rc;
}

#endif
