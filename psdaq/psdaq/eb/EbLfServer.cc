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


EbLfServer::EbLfServer(const char* addr,
                       const char* port) :
  _pep(nullptr)
{
  _status = _initialize(addr, port);
}

EbLfServer::~EbLfServer()
{
  if (_rxcq)  delete _rxcq;
  if (_pep)   delete _pep;
}

int EbLfServer::_initialize(const char* addr,
                            const char* port)
{
  const uint64_t flags  = 0;
  const size_t   txSize = 0;
  const size_t   rxSize = 300;
  _pep = new PassiveEndpoint(addr, port, flags, txSize, rxSize);
  if (!_pep || (_pep->state() != EP_UP))
  {
    fprintf(stderr, "%s: Failed to create Passive Endpoint: %s\n",
            __PRETTY_FUNCTION__, _pep ? _pep->error() : "No memory");
    return _pep ? _pep->error_num(): -FI_ENOMEM;
  }

  Fabric* fab = _pep->fabric();

  //void* data = fab;                     // Something since data can't be NULL
  //printf("EbLfServer is using LibFabric version '%s', fabric '%s', '%s' provider version %08x\n",
  //       fi_tostr(data, FI_TYPE_VERSION), fab->name(), fab->provider(), fab->version());

  _rxcq = new CompletionQueue(fab);
  if (!_rxcq)
  {
    fprintf(stderr, "%s: Failed to create RX completion queue: %s\n",
            __PRETTY_FUNCTION__, "No memory");
    return -FI_ENOMEM;
  }

  const int backlog = 64;
  if(!_pep->listen(backlog))
  {
    fprintf(stderr, "%s: Failed to set passive endpoint to listening state: %s\n",
            __PRETTY_FUNCTION__, _pep->error());
    return _pep->error_num();
  }
  printf("Listening for EbLfClient(s) on port %s\n", port);

  return 0;
}

int EbLfServer::connect(EbLfLink** link, int tmo)
{
  if (_status != 0)
  {
    fprintf(stderr, "%s: Failed to initialize Server\n", __PRETTY_FUNCTION__);
    return _status;
  }

  CompletionQueue* txcq    = nullptr;
  uint64_t         txFlags = 0;
  Endpoint* ep = _pep->accept(tmo, txcq, txFlags, _rxcq, FI_RECV);
  if (!ep)
  {
    fprintf(stderr, "%s: Failed to accept connection: %s\n",
            __PRETTY_FUNCTION__, _pep->error());
    return _pep->error_num();
  }

  int rxDepth = _pep->fabric()->info()->rx_attr->size;
  printf("rxDepth = %d\n", rxDepth);
  *link = new EbLfLink(ep, rxDepth);
  if (!*link)
  {
    fprintf(stderr, "%s: Failed to find memory for link\n", __PRETTY_FUNCTION__);
    return ENOMEM;
  }

  return 0;
}

int EbLfServer::_poll(fi_cq_data_entry* cqEntry, uint64_t flags)
{
  const int maxCnt = 1;
  //const int tmo    = 5000;              // milliseconds
  //ssize_t rc = _rxcq->comp_wait(cqEntry, maxCnt, tmo); // Waiting favors throughput
  ssize_t rc = _rxcq->comp(cqEntry, maxCnt);           // Polling favors latency
  if (rc == maxCnt)
  {
    if ((cqEntry->flags & flags) == flags)
    {
      //fprintf(stderr, "%s: Expected   CQ entry:\n"
      //                "  count %zd, got flags %016lx vs %016lx, data = %08lx\n"
      //                "  ctx   %p, len %zd, buf %p\n",
      //        __PRETTY_FUNCTION__, rc, cqEntry->flags, flags, cqEntry->data,
      //        cqEntry->op_context, cqEntry->len, cqEntry->buf);

      return 0;
    }

    fprintf(stderr, "%s: Unexpected CQ entry:\n"
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
      fprintf(stderr, "%s: Error reading RX completion queue: %s\n",
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

  const uint64_t flags = FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA;
  while ((rc = _poll(cqEntry, flags)) == -FI_EAGAIN)
  {
    auto t1(std::chrono::steady_clock::now());

    //const int tmo = 5000;               // milliseconds
    if (std::chrono::duration_cast<ms_t>(t1 - t0).count() > msTmo)
    {
      return -FI_ETIMEDOUT;
    }
  }

  return rc;
}

int EbLfServer::pend(void** ctx, int msTmo)
{
  fi_cq_data_entry cqEntry;

  int rc = pend(&cqEntry, msTmo);
  if (!rc)  *ctx = cqEntry.op_context;

  return rc;
}

int EbLfServer::pend(uint64_t* data, int msTmo)
{
  fi_cq_data_entry cqEntry;

  int rc = pend(&cqEntry, msTmo);
  if (!rc)  *data = cqEntry.data;

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
      fprintf(stderr, "%s: Unexpected event %u - expected FI_SHUTDOWN (%u)\n",
              __PRETTY_FUNCTION__, event, FI_SHUTDOWN);
      rc = ep->error_num();
    }
  }
  else
  {
    if (ep->error_num() != -FI_EAGAIN)
    {
      fprintf(stderr, "%s: Waiting for event failed: %s\n",
              __PRETTY_FUNCTION__, ep->error());
      rc = ep->error_num();
    }
  }

  _pep->close(ep);

  if (link)  delete link;

  return rc;
}
