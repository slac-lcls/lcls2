#include "EbLfServer.hh"

#include "EbLfLink.hh"
#include "Endpoint.hh"

#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <chrono>

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;

using ms_t = std::chrono::milliseconds;

static const int COMP_TMO = 5000;       // ms; Completion read timeout


EbLfServer::EbLfServer(unsigned verbose) :
  _tmo    (0),                          // Start by polling
  _verbose(verbose),
  _pending(0),
  _pep    (nullptr)
{
}

EbLfServer::~EbLfServer()
{
  if (_rxcq)  delete _rxcq;
  if (_pep)   delete _pep;
}

int EbLfServer::initialize(const std::string& addr,
                           const std::string& port)
{
  const uint64_t flags  = 0;
  const size_t   txSize = 0;
  const size_t   rxSize = 0;
  _pep = new PassiveEndpoint(addr.c_str(), port.c_str(), flags, txSize, rxSize);
  if (!_pep || (_pep->state() != EP_UP))
  {
    fprintf(stderr, "%s:\n  Failed to create Passive Endpoint: %s\n",
            __PRETTY_FUNCTION__, _pep ? _pep->error() : "No memory");
    return _pep ? _pep->error_num(): -FI_ENOMEM;
  }

  Fabric* fab = _pep->fabric();

  if (_verbose)
  {
    void* data = fab;                   // Something since data can't be NULL
    printf("EbLfServer is using LibFabric version '%s', fabric '%s', '%s' provider version %08x\n",
           fi_tostr(data, FI_TYPE_VERSION), fab->name(), fab->provider(), fab->version());
  }

  _rxcq = new CompletionQueue(fab);
  if (!_rxcq)
  {
    fprintf(stderr, "%s:\n  Failed to create RX completion queue: %s\n",
            __PRETTY_FUNCTION__, "No memory");
    return -FI_ENOMEM;
  }

  const int backlog = 64;
  if(!_pep->listen(backlog))
  {
    fprintf(stderr, "%s:\n  Failed to set passive endpoint to listening state: %s\n",
            __PRETTY_FUNCTION__, _pep->error());
    return _pep->error_num();
  }
  printf("EbLfServer is listening for client(s) on port %s\n", port.c_str());

  return 0;
}

int EbLfServer::connect(EbLfLink** link, int tmo)
{
  CompletionQueue* txcq    = nullptr;
  uint64_t         txFlags = 0;
  Endpoint* ep = _pep->accept(tmo, txcq, txFlags, _rxcq, FI_RECV);
  if (!ep)
  {
    fprintf(stderr, "%s:\n  Failed to accept connection: %s\n",
            __PRETTY_FUNCTION__, _pep->error());
    return _pep->error_num();
  }

  int rxDepth = _pep->fabric()->info()->rx_attr->size;
  *link = new EbLfLink(ep, rxDepth, _verbose, _unused);
  if (!*link)
  {
    fprintf(stderr, "%s:\n  Failed to find memory for link\n", __PRETTY_FUNCTION__);
    return ENOMEM;
  }

  return 0;
}

int EbLfServer::_poll(fi_cq_data_entry* cqEntry, uint64_t flags)
{
  // Polling favors latency, waiting favors throughput
  const int maxCnt = 1;
  ssize_t   rc     = _rxcq->comp_wait(cqEntry, maxCnt, _tmo);
  if (rc == maxCnt)
  {
    _tmo = 0;                 // Switch to polling after successful completion

    if ((cqEntry->flags & flags) == flags)
    {
      //fprintf(stderr, "%s:\n  Expected   CQ entry:\n"
      //                "  count %zd, got flags %016lx vs %016lx, data = %08lx\n"
      //                "  ctx   %p, len %zd, buf %p\n",
      //        __PRETTY_FUNCTION__, rc, cqEntry->flags, flags, cqEntry->data,
      //        cqEntry->op_context, cqEntry->len, cqEntry->buf);

      return 0;
    }

    fprintf(stderr, "%s:\n  Unexpected CQ entry:\n"
                    "  count %zd, got flags %016lx vs %016lx\n"
                    "  ctx   %p, len %zd, buf %p\n",
            __PRETTY_FUNCTION__, rc, cqEntry->flags, flags,
            cqEntry->op_context, cqEntry->len, cqEntry->buf);

    return -FI_EAGAIN;
  }

  // Man page suggests rc == 0 cannot occur
  if (rc != -FI_EAGAIN)
  {
    static int _errno = maxCnt;
    if (rc != _errno)
    {
      fprintf(stderr, "%s:\n  Error reading RX CQ: %s\n",
              __PRETTY_FUNCTION__, _rxcq->error());
      _errno = rc;
    }
  }

  return rc;
}

int EbLfServer::pend(fi_cq_data_entry* cqEntry, int msTmo)
{
  auto t0(std::chrono::steady_clock::now());
  int  rc;

  ++_pending;

  const uint64_t flags = FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA;
  while ((rc = _poll(cqEntry, flags)) == -FI_EAGAIN)
  {
    auto t1(std::chrono::steady_clock::now());

    if (std::chrono::duration_cast<ms_t>(t1 - t0).count() > msTmo)
    {
      _tmo = COMP_TMO;                  // Switch to waiting after a timeout
      rc   = -FI_ETIMEDOUT;
      break;
    }
  }

  --_pending;

  return rc;
}

int EbLfServer::pend(void** ctx, int msTmo)
{
  fi_cq_data_entry cqEntry;

  int rc = pend(&cqEntry, msTmo);
  *ctx = cqEntry.op_context;

  return rc;
}

int EbLfServer::pend(uint64_t* data, int msTmo)
{
  fi_cq_data_entry cqEntry;

  int rc = pend(&cqEntry, msTmo);
  *data = cqEntry.data;

  return rc;
}

int EbLfServer::poll(uint64_t* data)
{
  const uint64_t   flags = FI_MSG | FI_RECV | FI_REMOTE_CQ_DATA;
  fi_cq_data_entry cqEntry;

  int rc = _poll(&cqEntry, flags);
  if (!rc)  *data = cqEntry.data;

  return rc;
}

int EbLfServer::shutdown(EbLfLink* link)
{
  int rc = FI_SUCCESS;

  Endpoint* ep = link->endpoint();
  if (!ep)  return -1;

  bool cm_entry;
  struct fi_eq_cm_entry entry;
  uint32_t event;
  const int tmo = 1000;               // mS
  if (ep->event_wait(&event, &entry, &cm_entry, tmo))
  {
    if (cm_entry && (event == FI_SHUTDOWN))
    {
      printf("EbLfClient %d disconnected\n", link->id());
    }
    else
    {
      fprintf(stderr, "%s:\n  Unexpected event %u - expected FI_SHUTDOWN (%u)\n",
              __PRETTY_FUNCTION__, event, FI_SHUTDOWN);
      rc = ep->error_num();
    }
  }
  else
  {
    if (ep->error_num() != -FI_EAGAIN)
    {
      fprintf(stderr, "%s:\n  Waiting for event failed: %s\n",
              __PRETTY_FUNCTION__, ep->error());
      rc = ep->error_num();
    }
  }

  _pep->close(ep);

  if (link)  delete link;

  return rc;
}
