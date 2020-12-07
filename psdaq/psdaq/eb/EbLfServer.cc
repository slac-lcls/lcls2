#include "EbLfServer.hh"

#include "Endpoint.hh"

#include "psdaq/service/fast_monotonic_clock.hh"

#include <chrono>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;
using ms_t = std::chrono::milliseconds;


EbLfServer::EbLfServer(const unsigned& verbose) :
  _eq     (nullptr),
  _rxcq   (nullptr),
  _tmo    (0),                          // Start by polling
  _verbose(verbose),
  _pending(0),
  _pep    (nullptr)
{
}

EbLfServer::EbLfServer(const unsigned&                           verbose,
                       const std::map<std::string, std::string>& kwargs) :
  _eq     (nullptr),
  _rxcq   (nullptr),
  _tmo    (0),                          // Start by polling
  _verbose(verbose),
  _pending(0),
  _pep    (nullptr),
  _info   (kwargs)
{
}

EbLfServer::~EbLfServer()
{
  shutdown();
}

int EbLfServer::listen(const std::string& addr,
                       std::string&       port,
                       unsigned           nLinks)
{
  if (!_info.ready()) {
    fprintf(stderr, "%s:\n  Failed to set up Info structure: %s\n",
            __PRETTY_FUNCTION__, _info.error());
    return _info.error_num();
  }

  _pending = 0;

  const uint64_t flags = 0;               // For fi_getinfo(), e.g., FI_SOURCE
  _info.hints->tx_attr->size = 0;         // Default for the return path
  _info.hints->rx_attr->size = 0; //1152 + 64; // Tunable parameter
  _pep = new PassiveEndpoint(addr.c_str(), port.c_str(), flags, &_info);
  if (!_pep || (_pep->state() != EP_UP))
  {
    fprintf(stderr, "%s:\n  Failed to create Passive Endpoint for %s:%s: %s\n",
            __PRETTY_FUNCTION__, addr.c_str(), port.c_str(), _pep ? _pep->error() : "No memory");
    return _pep ? _pep->error_num(): ENOMEM;
  }

  Fabric* fab = _pep->fabric();

  if (_verbose)
  {
    void* data = fab;                   // Something since data can't be NULL
    printf("Server: LibFabric version '%s', domain '%s', fabric '%s', provider '%s', version %08x\n",
           fi_tostr(data, FI_TYPE_VERSION), fab->domain_name(), fab->fabric_name(), fab->provider(), fab->version());
  }

  _eq = new EventQueue(fab, 0);
  if (!_eq)
  {
    fprintf(stderr, "%s:\n  Failed to create Event Queue: %s\n",
            __PRETTY_FUNCTION__, "No memory");
    return ENOMEM;
  }

  struct fi_info* info   = fab->info();
  size_t          cqSize = nLinks * info->rx_attr->size;
  if (_verbose > 1)  printf("EbLfServer: rx_attr.size = %zd, tx_attr.size = %zd\n",
                            info->rx_attr->size, info->tx_attr->size);
  _rxcq = new CompletionQueue(fab, cqSize);
  if (!_rxcq)
  {
    fprintf(stderr, "%s:\n  Failed to create Rx Completion Queue: %s\n",
            __PRETTY_FUNCTION__, "No memory");
    return ENOMEM;
  }

  bool rc;
  if (port.empty())                     // OS provides an ephemeral port number
  {
    uint16_t newPort;
    rc   = _pep->listen(nLinks, newPort);
    port = std::to_string(unsigned(newPort));
  }
  else                                  // Use the caller-provided port number
    rc = _pep->listen(nLinks);
  if (!rc)
  {
    fprintf(stderr, "%s:\n  Failed to set Passive Endpoint to listening state: %s\n",
            __PRETTY_FUNCTION__, _pep->error());
    return _pep->error_num();
  }

  if (_verbose)
    printf("EbLfServer is listening for up to %u client(s) on %s:%s\n",
           nLinks, addr.c_str(), port.c_str());

  return 0;
}

int EbLfServer::connect(EbLfSvrLink** link, int msTmo)
{
  auto             t0(std::chrono::steady_clock::now());
  CompletionQueue* txcq    = nullptr;
  uint64_t         txFlags = 0;
  uint64_t         rxFlags = FI_RECV;
  Endpoint* ep = _pep->accept(msTmo, _eq, txcq, txFlags, _rxcq, rxFlags);
  if (!ep)
  {
    fprintf(stderr, "%s:\n  Failed to accept connection: %s\n",
            __PRETTY_FUNCTION__, _pep->error());
    return _pep->error_num();
  }
  auto t1(std::chrono::steady_clock::now());
  auto dT(std::chrono::duration_cast<ms_t>(t1 - t0).count());
  if (_verbose > 1)  printf("EbLfServer: dT to connect: %lu ms\n", dT);

  int rxDepth = ep->fabric()->info()->rx_attr->size;
  if (_verbose > 1)  printf("EbLfServer: rx_attr.size = %d\n", rxDepth);
  *link = new EbLfSvrLink(ep, rxDepth, _verbose);
  if (!*link)
  {
    fprintf(stderr, "%s:\n  Failed to find memory for link\n", __PRETTY_FUNCTION__);
    delete ep;
    return ENOMEM;
  }
  _linkByEp[ep->endpoint()] = *link;

  return 0;
}

int EbLfServer::setupMr(void* region, size_t size)
{
  if (_pep)
    return Pds::Eb::setupMr(_pep->fabric(), region, size, nullptr, _verbose);
  else
    return -1;
}

int EbLfServer::pollEQ()
{
  int rc;

  bool                  cmEntry;
  struct fi_eq_cm_entry entry;
  uint32_t              event;

  if (_eq->event(&event, &entry, &cmEntry))
  {
    if (cmEntry && (event == FI_SHUTDOWN))
    {
      fid_ep* ep = reinterpret_cast<fid_ep*>(entry.fid);
      if (_linkByEp.find(ep) != _linkByEp.end())
      {
        EbLfSvrLink* link = _linkByEp[ep];
        if (_verbose)
          printf("EbLfClient %d disconnected\n", link->id());
        _linkByEp.erase(ep);
        rc = (_linkByEp.size() == 0) ? -FI_ENOTCONN : FI_SUCCESS;
      }
      else
      {
        fprintf(stderr, "%s:\n  Ignoring unrecognized EP %p "
                "during FI_SHUTDOWN event\n", __PRETTY_FUNCTION__, ep);
        rc = -FI_ENOKEY;
      }
    }
    else
    {
      fid_ep* ep = cmEntry ? reinterpret_cast<fid_ep*>(entry.fid) : nullptr;
      if (_linkByEp.find(ep) != _linkByEp.end())
      {
        fprintf(stderr, "%s:\n  Unexpected event %u from EbLfClient %d\n",
                __PRETTY_FUNCTION__, event, _linkByEp[ep]->id());
      }
      else
      {
        fprintf(stderr, "%s:\n  Ignoring unrecognized EP %p "
                "during unexpected event %u\n",
                __PRETTY_FUNCTION__, ep, event);
      }
      rc = _eq->error_num();
    }
  }
  else
  {
    if (_eq->error_num() != -FI_EAGAIN)
    {
      fprintf(stderr, "%s:\n  Failed to read from Event Queue: %s\n",
              __PRETTY_FUNCTION__, _eq->error());
      rc = _eq->error_num();
    }
  }

  return rc;
}

int EbLfServer::disconnect(EbLfSvrLink* link)
{
  if (!link)  return FI_SUCCESS;

  if (_verbose)
    printf("Disconnecting from EbLfClient %d\n", link->id());

  Endpoint* ep = link->endpoint();
  if (!ep)  return -FI_ENOTCONN;

  ep->shutdown();

  delete link;

  return FI_SUCCESS;
}

void EbLfServer::shutdown()
{
  if (_pep)
  {
    delete _pep;
    _pep = nullptr;
  }
  if (_rxcq)
  {
    delete _rxcq;
    _rxcq = nullptr;
  }
  if (_eq)
  {
    delete _eq;
    _eq = nullptr;
  }
}

int EbLfServer::pend(fi_cq_data_entry* cqEntry, int msTmo)
{
  int                              rc;
  fast_monotonic_clock::time_point t0;
  bool                             first = true;

  ++_pending;

  while (true)
  {
    const uint64_t flags = FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA;
    rc = _poll(cqEntry, flags);
    if (rc > 0)
    {
      break;
    }
    else if (rc == -FI_EAGAIN)
    {
      if (_tmo)
      {
        rc = -FI_ETIMEDOUT;
        break;
      }
      if (!first)
      {
        using ms_t = std::chrono::milliseconds;
        auto  t1   = fast_monotonic_clock::now();

        if (std::chrono::duration_cast<ms_t>(t1 - t0).count() > msTmo)
        {
          _tmo = msTmo;               // Switch to waiting after a timeout
          rc = -FI_ETIMEDOUT;
          break;
        }
      }
      else
      {
        t0    = fast_monotonic_clock::now();
        first = false;
      }
    }
    else
    {
      static int _errno = 1;
      if (rc != _errno)
      {
        fprintf(stderr, "%s:\n  Error reading Rx CQ: %s(%d)\n",
                __PRETTY_FUNCTION__, _rxcq->error(), rc);
        _errno = rc;
      }
      break;
    }
  }

  --_pending;

  return rc;
}


// --- Revisit: The following maybe better belongs somewhere else
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;

int Pds::Eb::linksStart(EbLfServer&        transport,
                        const std::string& ifAddr,
                        std::string&       port,
                        unsigned           nLinks,
                        const char*        peer)
{
  int rc = transport.listen(ifAddr, port, nLinks);
  if (rc)
  {
    logging::error("%s:\n  Failed to initialize %s EbLfServer on %s:%s",
                   __PRETTY_FUNCTION__, peer, ifAddr.c_str(), port.c_str());
    return rc;
  }

  return 0;
}

int Pds::Eb::linksConnect(EbLfServer&                transport,
                          std::vector<EbLfSvrLink*>& links,
                          const char*                peer)
{
  for (unsigned i = 0; i < links.size(); ++i)
  {
    auto           t0(std::chrono::steady_clock::now());
    int            rc;
    EbLfSvrLink*   link;
    const unsigned msTmo(14750);        // < control.py transition timeout
    if ( (rc = transport.connect(&link, msTmo)) )
    {
      logging::error("%s:\n  Error connecting to a %s",
                     __PRETTY_FUNCTION__, peer);
      return rc;
    }
    links[i] = link;

    auto t1 = std::chrono::steady_clock::now();
    auto dT = std::chrono::duration_cast<ms_t>(t1 - t0).count();
    logging::info("Inbound link[%u] with %s connected in %lu ms",
                  i, peer, dT);
  }

  return 0;
}

int Pds::Eb::linksConfigure(std::vector<EbLfSvrLink*>& links,
                            unsigned                   id,
                            const char*                peer)
{
  std::vector<EbLfSvrLink*> tmpLinks(links.size());

  for (auto link : links)
  {
    auto t0(std::chrono::steady_clock::now());
    int  rc;
    if ( (rc = link->prepare(id, peer)) )
    {
      logging::error("%s:\n  Failed to prepare link with %s ID %d",
                     __PRETTY_FUNCTION__, peer, link->id());
      return rc;
    }
    unsigned rmtId  = link->id();
    tmpLinks[rmtId] = link;

    auto t1 = std::chrono::steady_clock::now();
    auto dT = std::chrono::duration_cast<ms_t>(t1 - t0).count();
    logging::info("Inbound link with %s ID %d configured in %lu ms",
                  peer, rmtId, dT);
  }

  links = tmpLinks;                     // Now in remote ID sorted order

  return 0;
}
