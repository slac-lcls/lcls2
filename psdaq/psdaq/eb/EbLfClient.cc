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


EbLfClient::EbLfClient(const unsigned& verbose) :
  _pending(0),
  _verbose(verbose)
{
}

EbLfClient::EbLfClient(const unsigned&                           verbose,
                       const std::map<std::string, std::string>& kwargs) :
  _pending(0),
  _verbose(verbose),
  _info   (kwargs)
{
}

int EbLfClient::connect(EbLfCltLink** link,
                        const char*   peer,
                        const char*   port,
                        unsigned      msTmo)
{
  if (!_info.ready()) {
    fprintf(stderr, "%s:\n  Failed to set up Info structure: %s\n",
            __PRETTY_FUNCTION__, _info.error());
    return _info.error_num();
  }

  _pending = 0;

  const uint64_t flags  = 0;
  _info.hints->tx_attr->size = 0; //192;     // Tunable parameter
  _info.hints->rx_attr->size = 0;       // Default for the return path
  Fabric* fab = new Fabric(peer, port, flags, &_info);
  if (!fab || !fab->up())
  {
    fprintf(stderr, "%s:\n  Failed to create Fabric for %s:%s: %s\n",
            __PRETTY_FUNCTION__, peer, port, fab ? fab->error() : "No memory");
    return fab ? fab->error_num() : -FI_ENOMEM;
  }

  if (_verbose)
  {
    void* data = fab;                   // Something since data can't be NULL
    printf("Client: LibFabric version '%s', domain '%s', fabric '%s', provider '%s', version %08x\n",
           fi_tostr(data, FI_TYPE_VERSION), fab->domain_name(), fab->fabric_name(), fab->provider(), fab->version());
  }

  struct fi_info*  info   = fab->info();
  size_t           cqSize = info->tx_attr->size;
  if (_verbose > 1)  printf("EbLfClient: tx_attr.size = %zd\n", cqSize);
  CompletionQueue* txcq   = new CompletionQueue(fab, cqSize);
  if (!txcq)
  {
    fprintf(stderr, "%s:\n  Failed to create Tx completion queue: %s\n",
            __PRETTY_FUNCTION__, "No memory");
    return -FI_ENOMEM;
  }

  if (_verbose)
    printf("EbLfClient is waiting %u ms for server %s:%s\n", msTmo, peer, port);

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
    uint64_t txFlags = FI_TRANSMIT | FI_SELECTIVE_COMPLETION;
    uint64_t rxFlags = 0;
    if (ep->connect(msTmo, txFlags, rxFlags))  break; // Success
    if (ep->error_num() == -FI_ENODATA)        break; // connect() timed out
    if (ep->error_num() != -FI_ECONNREFUSED)   break; // Serious error

    t1 = std::chrono::steady_clock::now();
    dT = std::chrono::duration_cast<ms_t>(t1 - t0).count();
    if (msTmo && (dT > msTmo))  break;

    ep->shutdown();                 // Can't try to connect on an EP a 2nd time

    // Retrying too quickly can cause libfabric sockets EP to segfault
    // Jan 2020: LF 1.7.1, sock_ep_cm_thread()->sock_pep_req_handler(), pep = 0
    std::this_thread::sleep_for(ms_t(1));
  }
  t1 = std::chrono::steady_clock::now();
  dT = std::chrono::duration_cast<ms_t>(t1 - t0).count();
  if ((ep->error_num() != FI_SUCCESS) || (msTmo && (dT > msTmo)))
  {
    int rc = ep->error_num();
    fprintf(stderr, "%s:\n  %s connecting to %s:%s: %s\n",
            __PRETTY_FUNCTION__, (dT <= msTmo) ? "Error" : "Timed out",
            peer, port, ep->error());
    delete ep;
    return (dT <= msTmo) ? rc : -FI_ETIMEDOUT;
  }
  if (_verbose > 1)  printf("EbLfClient: dT to connect: %lu ms\n", dT);

  int rxDepth = fab->info()->rx_attr->size;
  if (_verbose > 1)  printf("EbLfClient: rx_attr.size = %d\n", rxDepth);
  *link = new EbLfCltLink(ep, rxDepth, _verbose, _pending);
  if (!*link)
  {
    fprintf(stderr, "%s:\n  Failed to find memory for link\n", __PRETTY_FUNCTION__);
    delete ep;
    return ENOMEM;
  }

  return 0;
}

int EbLfClient::disconnect(EbLfCltLink* link)
{
  if (!link)  return FI_SUCCESS;

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

  return FI_SUCCESS;
}


// --- Revisit: The following maybe better belongs somewhere else
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;

int Pds::Eb::linksConnect(EbLfClient&                     transport,
                          std::vector<EbLfCltLink*>&      links,
                          const std::vector<std::string>& addrs,
                          const std::vector<std::string>& ports,
                          const char*                     peer)
{
  for (unsigned i = 0; i < addrs.size(); ++i)
  {
    auto           t0(std::chrono::steady_clock::now());
    int            rc;
    const char*    addr = addrs[i].c_str();
    const char*    port = ports[i].c_str();
    EbLfCltLink*   link;
    const unsigned msTmo(14750);        // < control.py transition timeout
    if ( (rc = transport.connect(&link, addr, port, msTmo)) )
    {
      logging::error("%s:\n  Error connecting to %s at %s:%s",
                     __PRETTY_FUNCTION__, peer, addr, port);
      return rc;
    }
    links[i] = link;

    auto t1 = std::chrono::steady_clock::now();
    auto dT = std::chrono::duration_cast<ms_t>(t1 - t0).count();
    logging::info("Outbound link[%u] with %s connected in %lu ms",
                  i, peer, dT);
  }

  return 0;
}

int Pds::Eb::linksConfigure(std::vector<EbLfCltLink*>& links,
                            unsigned                   id,
                            const char*                peer)
{
  return linksConfigure(links, id, nullptr, 0, 0, peer);
}

int Pds::Eb::linksConfigure(std::vector<EbLfCltLink*>& links,
                            unsigned                   id,
                            void*                      region,
                            size_t                     regSize,
                            const char*                peer)
{
  return linksConfigure(links, id, region, regSize, regSize, peer);
}

int Pds::Eb::linksConfigure(std::vector<EbLfCltLink*>& links,
                            unsigned                   id,
                            void*                      region,
                            size_t                     lclSize,
                            size_t                     rmtSize,
                            const char*                peer)
{
  std::vector<EbLfCltLink*> tmpLinks(links.size());

  for (auto link : links)
  {
    auto t0(std::chrono::steady_clock::now());
    int  rc = link->prepare(id, region, lclSize, rmtSize, peer);
    if (rc)
    {
      logging::error("%s:\n  Failed to prepare link with %s ID %d",
                     __PRETTY_FUNCTION__, peer, link->id());
      return rc;
    }
    unsigned rmtId  = link->id();
    tmpLinks[rmtId] = link;

    auto t1 = std::chrono::steady_clock::now();
    auto dT = std::chrono::duration_cast<ms_t>(t1 - t0).count();
    logging::info("Outbound link with %s ID %d configured in %lu ms",
                  peer, rmtId, dT);
  }

  links = tmpLinks;                     // Now in remote ID sorted order

  return 0;
}
