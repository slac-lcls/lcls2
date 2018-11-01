#include "EbLfClient.hh"

#include "EbLfLink.hh"
#include "Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>                     // For sleep()...
#include <assert.h>
#include <chrono>

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;

using ms_t = std::chrono::milliseconds;


EbLfClient::EbLfClient()
{
}

EbLfClient::~EbLfClient()
{
}

int EbLfClient::connect(const char* peer,
                        const char* port,
                        unsigned    tmo,
                        EbLfLink**  link)
{
  const uint64_t flags  = 0;
  const size_t   txSize = 0;
  const size_t   rxSize = 0;
  Fabric* fab = new Fabric(peer, port, flags, txSize, rxSize);
  if (!fab || !fab->up())
  {
    fprintf(stderr, "%s: Failed to create Fabric for %s:%s: %s\n",
            __PRETTY_FUNCTION__, peer, port, fab ? fab->error() : "No memory");
    return fab ? fab->error_num() : -FI_ENOMEM;
  }

  //void* data = fab;                     // Something since data can't be NULL
  //printf("EbLfClient is using LibFabric version '%s', fabric '%s', '%s' provider version %08x\n",
  //       fi_tostr(data, FI_TYPE_VERSION), fab->name(), fab->provider(), fab->version());

  CompletionQueue* txcq = new CompletionQueue(fab);
  if (!txcq)
  {
    fprintf(stderr, "%s: Failed to create TX completion queue: %s\n",
            __PRETTY_FUNCTION__, "No memory");
    return -FI_ENOMEM;
  }

  printf("Waiting for EbLfServer %s:%s\n", peer, port);

  Endpoint* ep         = nullptr;
  bool      tmoEnabled = tmo != 0;
  int       timeout    = tmoEnabled ? tmo : -1; // mS
  auto      t0(std::chrono::steady_clock::now());
  auto      t1(t0);
  uint64_t  dT = 0;
  while (true)
  {
    CompletionQueue* rxcq = nullptr;
    ep = new Endpoint(fab, txcq, rxcq);
    if (!ep || (ep->state() != EP_UP))
    {
      fprintf(stderr, "%s: Failed to initialize Endpoint: %s\n",
              __PRETTY_FUNCTION__, ep ? ep->error() : "No memory");
      return ep ? ep->error_num() : -FI_ENOMEM;
    }

    if (ep->connect(timeout, FI_TRANSMIT | FI_SELECTIVE_COMPLETION, 0))  break;
    if (ep->error_num() == -FI_ENODATA)  break; // connect() timed out

    t1 = std::chrono::steady_clock::now();
    dT = std::chrono::duration_cast<ms_t>(t1 - t0).count();
    if (tmoEnabled && (dT > tmo))  break;

    delete ep;                      // Can't try to connect on an EP a 2nd time

    usleep(100000);
  }
  if ((ep->error_num() != FI_SUCCESS) || (tmoEnabled && (dT > tmo)))
  {
    int rc = ep->error_num();
    fprintf(stderr, "%s: Error connecting to %s:%s: %s\n",
            __PRETTY_FUNCTION__, peer, port,
            (rc == FI_SUCCESS) ? ep->error() : "Timed out");
    delete ep;
    return (rc != FI_SUCCESS) ? rc : -FI_ETIMEDOUT;
  }

  *link = new EbLfLink(ep);
  if (!*link)
  {
    fprintf(stderr, "%s: Failed to find memory for link\n", __PRETTY_FUNCTION__);
    return ENOMEM;
  }

  return 0;
}

int EbLfClient::shutdown(EbLfLink* link)
{
  if (link)
  {
    Endpoint* ep = link->endpoint();
    delete link;
    if (ep)
    {
      CompletionQueue* txcq = ep->txcq();
      Fabric*          fab  = ep->fabric();
      delete ep;
      if (txcq)  delete txcq;
      if (fab)   delete fab;
    }
  }

  return 0;
}
