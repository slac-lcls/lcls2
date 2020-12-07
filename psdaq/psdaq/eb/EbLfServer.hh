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

    using LinkMap = std::unordered_map<fid_ep*, EbLfSvrLink*>;

    class EbLfServer
    {
    public:
      EbLfServer(const unsigned& verbose);
      EbLfServer(const unsigned&                           verbose,
                 const std::map<std::string, std::string>& kwargs);
      ~EbLfServer();
    public:
      int  listen(const std::string& addr,    // Interface to use
                  std::string&       port,    // Port being listened on
                  unsigned           nLinks); // Max number of links
      int  connect(EbLfSvrLink**, int msTmo = -1);
      int  disconnect(EbLfSvrLink*);
      void shutdown();
      int  pend(fi_cq_data_entry*, int msTmo);
      int  pend(void** context, int msTmo);
      int  pend(uint64_t* data, int msTmo);
      int  poll(uint64_t* data);
      int  pollEQ();
      int  setupMr(void* region, size_t size);
    public:
      const uint64_t& pending() const { return *const_cast<uint64_t*>(&_pending); } // Cast away volatile
    private:
      int _poll(fi_cq_data_entry*, uint64_t flags);
    private:                              // Arranged in order of access frequency
      Fabrics::EventQueue*      _eq;      // Event Queue
      Fabrics::CompletionQueue* _rxcq;    // Receive Completion Queue
      int                       _tmo;     // Timeout for polling or waiting
      const unsigned&           _verbose; // Print some stuff if set
    private:
      uint64_t                  _pending; // Flag set when currently pending
    private:
      Fabrics::PassiveEndpoint* _pep;     // EP for establishing connections
      LinkMap                   _linkByEp;// Map to retrieve link given raw EP
      Fabrics::Info             _info;    // Connection options
    };

    // --- Revisit: The following maybe better belongs somewhere else

    int linksStart(EbLfServer&        transport,
                   const std::string& ifAddr,
                   std::string&       port,
                   unsigned           nLinks,
                   const char*        name);
    int linksConnect(EbLfServer&                transport,
                     std::vector<EbLfSvrLink*>& links,
                     const char*                name);
    int linksConfigure(std::vector<EbLfSvrLink*>& links,
                       unsigned                   id,
                       const char*                name);

  };
};

inline
int Pds::Eb::EbLfServer::_poll(fi_cq_data_entry* cqEntry, uint64_t flags)
{
  ssize_t rc;

  // Polling favors latency, waiting favors throughput
  if (!_tmo)
  {
    rc = _rxcq->comp(cqEntry, 1); // Uses much less kernel time than comp_wait() with tmo = 0
  }
  else
  {
    rc = _rxcq->comp_wait(cqEntry, 1, _tmo);
    if (rc > 0)  _tmo = 0;     // Switch to polling after successful completion
  }

  if (rc > 0)
  {
    if (cqEntry->op_context)
    {
      auto link = static_cast<Pds::Eb::EbLfLink*>(cqEntry->op_context);
      link->_credits -= rc;
      if (link->_credits < 0)
        printf("*** 3 link %p credits (%ld) < 0\n", link, link->_credits);
      link->postCompRecv(rc);
    }
    //else
    //  printf("cqEntry->op_context is NULL\n");

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

inline
int Pds::Eb::EbLfServer::poll(uint64_t* data)
{
  const uint64_t   flags = FI_MSG | FI_RECV | FI_REMOTE_CQ_DATA;
  fi_cq_data_entry cqEntry;

  int rc = _poll(&cqEntry, flags);
  *data = cqEntry.data;

  return rc;
}

#endif
