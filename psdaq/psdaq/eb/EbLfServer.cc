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


EbLfServer::EbLfServer(unsigned verbose) :
  _eq     (nullptr),
  _rxcq   (nullptr),
  _tmo    (0),                          // Start by polling
  _verbose(verbose),
  _pending(0),
  _pep    (nullptr)
{
}

EbLfServer::~EbLfServer()
{
  shutdown();
}

int EbLfServer::initialize(const std::string& addr,
                           const std::string& port,
                           unsigned           nLinks)
{
  _pending = 0;

  const uint64_t flags  = 0;
  const size_t   txSize = 1;            // Something small to not waste memory
  const size_t   rxSize = 0; //1152 + 64;
  _pep = new PassiveEndpoint(addr.c_str(), port.c_str(), flags, txSize, rxSize);
  if (!_pep || (_pep->state() != EP_UP))
  {
    fprintf(stderr, "%s:\n  Failed to create Passive Endpoint: %s\n",
            __PRETTY_FUNCTION__, _pep ? _pep->error() : "No memory");
    return _pep ? _pep->error_num(): ENOMEM;
  }

  Fabric* fab = _pep->fabric();

  if (_verbose > 1)
  {
    void* data = fab;                   // Something since data can't be NULL
    printf("EbLfServer is using LibFabric version '%s', fabric '%s', '%s' provider version %08x\n",
           fi_tostr(data, FI_TYPE_VERSION), fab->name(), fab->provider(), fab->version());
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

  if (!_pep->listen(nLinks))
  {
    fprintf(stderr, "%s:\n  Failed to set Passive Endpoint to listening state: %s\n",
            __PRETTY_FUNCTION__, _pep->error());
    return _pep->error_num();
  }

  if (_verbose)
    printf("EbLfServer is listening for %d client(s) on port %s\n",
           nLinks, port.c_str());

  return 0;
}

int EbLfServer::connect(EbLfSvrLink** link, unsigned id, int msTmo)
{
  CompletionQueue* txcq    = nullptr;
  uint64_t         txFlags = 0;
  Endpoint* ep = _pep->accept(msTmo, _eq, txcq, txFlags, _rxcq, FI_RECV);
  if (!ep)
  {
    fprintf(stderr, "%s:\n  Failed to accept connection: %s\n",
            __PRETTY_FUNCTION__, _pep->error());
    return _pep->error_num();
  }

  int rxDepth = _pep->fabric()->info()->rx_attr->size;
  if (_verbose > 1)  printf("EbLfServer: rx_attr.size = %d\n", rxDepth);
  *link = new EbLfSvrLink(ep, rxDepth, _verbose);
  if (!*link)
  {
    fprintf(stderr, "%s:\n  Failed to find memory for link\n", __PRETTY_FUNCTION__);
    return ENOMEM;
  }
  _linkByEp[ep->endpoint()] = *link;

  int rc = (*link)->exchangeIds(id);
  if (rc)
  {
    fprintf(stderr, "%s:\n  Failed to exchange ID with peer\n", __PRETTY_FUNCTION__);
    return rc;
  }

  return 0;
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
                "during unexpected event %d\n",
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
  if (_verbose)
    printf("Disconnecting from EbLfClient %d\n", link->id());

  Endpoint* ep = link->endpoint();
  if (!ep)  return -FI_ENOTCONN;

  _pep->close(ep);

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
        fprintf(stderr, "%s:\n  Error reading Rx CQ: %s\n",
                __PRETTY_FUNCTION__, _rxcq->error());
        _errno = rc;
      }
      break;
    }
  }

  --_pending;

  return rc;
}
