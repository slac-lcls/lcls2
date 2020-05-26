#include "EbLfClient.hh"

#include "Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <thread>

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;

using ms_t = std::chrono::milliseconds;


EbLfClient::EbLfClient(unsigned verbose) :
  _pending(0),
  _verbose(verbose)
{
}

int EbLfClient::connect(EbLfCltLink** link,
                        const char*   peer,
                        const char*   port,
                        unsigned      id,
                        unsigned      msTmo)
{
  _pending = 0;

  const uint64_t flags  = 0;
  const size_t   txSize = 0; //192;
  const size_t   rxSize = 1;            // Something small to not waste memory
  Fabric* fab = new Fabric(peer, port, flags, txSize, rxSize);
  if (!fab || !fab->up())
  {
    fprintf(stderr, "%s:\n  Failed to create Fabric for %s:%s: %s\n",
            __PRETTY_FUNCTION__, peer, port, fab ? fab->error() : "No memory");
    return fab ? fab->error_num() : -FI_ENOMEM;
  }

  if (_verbose)
  {
    void* data = fab;                   // Something since data can't be NULL
    printf("LibFabric version '%s', fabric '%s', '%s' provider version %08x\n",
           fi_tostr(data, FI_TYPE_VERSION), fab->name(), fab->provider(), fab->version());
  }

  struct fi_info*  info   = fab->info();
  size_t           cqSize = info->tx_attr->size;
  if (_verbose > 1)  printf("EbLfClient: rx_attr.size = %zd, tx_attr.size = %zd\n",
                            info->rx_attr->size, info->tx_attr->size);
  CompletionQueue* txcq   = new CompletionQueue(fab, cqSize);
  if (!txcq)
  {
    fprintf(stderr, "%s:\n  Failed to create Tx completion queue: %s\n",
            __PRETTY_FUNCTION__, "No memory");
    return -FI_ENOMEM;
  }

  if (_verbose)
    printf("EbLfClient is waiting %d ms for server %s:%s\n", msTmo, peer, port);

  EventQueue*      eq   = nullptr;
  CompletionQueue* rxcq = nullptr;
  Endpoint*        ep   = new Endpoint(fab, eq, txcq, rxcq);
  if (!ep || (ep->state() != EP_UP))
  {
    fprintf(stderr, "%s:\n  Failed to initialize Endpoint: %s\n",
            __PRETTY_FUNCTION__, ep ? ep->error() : "No memory");
    return ep ? ep->error_num() : -FI_ENOMEM;
  }

  auto     t0(std::chrono::steady_clock::now());
  auto     t1(t0);
  uint64_t dT = 0;
  while (true)
  {
    if (ep->connect(msTmo, FI_TRANSMIT | FI_SELECTIVE_COMPLETION, 0))  break; // Success
    if (ep->error_num() == -FI_ENODATA)       break; // connect() timed out
    if (ep->error_num() != -FI_ECONNREFUSED)  break; // Serious error

    t1 = std::chrono::steady_clock::now();
    dT = std::chrono::duration_cast<ms_t>(t1 - t0).count();
    if (msTmo && (dT > msTmo))  break;

    ep->shutdown();                 // Can't try to connect on an EP a 2nd time

    // Retrying too quickly can cause libfabric sockets EP to segfault
    // Jan 2020: LF 1.7.1, sock_ep_cm_thread()->sock_pep_req_handler(), pep = 0
    std::this_thread::sleep_for(ms_t(100));
  }
  if ((ep->error_num() != FI_SUCCESS) || (msTmo && (dT > msTmo)))
  {
    int rc = ep->error_num();
    fprintf(stderr, "%s:\n  Error connecting to %s:%s: %s\n",
            __PRETTY_FUNCTION__, peer, port,
            (rc != FI_ENODATA) ? ep->error() : "Timed out");
    return (rc != FI_SUCCESS) ? rc : -FI_ETIMEDOUT;
  }

  *link = new EbLfCltLink(ep, _verbose, _pending);
  if (!*link)
  {
    fprintf(stderr, "%s:\n  Failed to find memory for link\n", __PRETTY_FUNCTION__);
    return ENOMEM;
  }

  int rc = (*link)->exchangeIds(id);
  if (rc)
  {
    fprintf(stderr, "%s:\n  Failed to exchange ID with peer\n", __PRETTY_FUNCTION__);
    return rc;
  }

  return 0;
}

int EbLfClient::disconnect(EbLfCltLink* link)
{
  if (_verbose)
    printf("Disconnecting from EbLfServer %d\n", link->id());

  Endpoint* ep = link->endpoint();
  if (ep)
  {
    CompletionQueue* txcq = ep->txcq();
    Fabric*          fab  = ep->fabric();
    if (txcq)  delete txcq;
    if (fab)   delete fab;
    delete ep;
  }
  delete link;
  _pending = 0;

  return 0;
}
