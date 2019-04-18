#include "EbLfServer.hh"

#include "EbLfLink.hh"
#include "Endpoint.hh"

#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;


static const int COMP_TMO = 5000;       // ms; Completion read timeout


EbLfServer::EbLfServer(unsigned verbose) :
  _eq     (nullptr),
  _rxcq   (nullptr),
  _tmo    (0),                          // Start by polling
  _verbose(verbose),
  _pending(0),
  _pep    (nullptr),
  _linkByEp()
{
}

int EbLfServer::initialize(const std::string& addr,
                           const std::string& port,
                           unsigned           nLinks)
{
  _pending = 0;

  const uint64_t flags  = 0;
  const size_t   txSize = 1;            // Something small to not waste memory
  const size_t   rxSize = 1152 + 64;
  _pep = new PassiveEndpoint(addr.c_str(), port.c_str(), flags, txSize, rxSize);
  if (!_pep || (_pep->state() != EP_UP))
  {
    fprintf(stderr, "%s:\n  Failed to create Passive Endpoint: %s\n",
            __PRETTY_FUNCTION__, _pep ? _pep->error() : "No memory");
    return _pep ? _pep->error_num(): ENOMEM;
  }

  Fabric* fab = _pep->fabric();

  if (_verbose)
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
  //printf("rx_attr.size = %zd, tx_attr.size = %zd\n",
  //       info->rx_attr->size, info->tx_attr->size);
  _rxcq = new CompletionQueue(fab, cqSize);
  if (!_rxcq)
  {
    fprintf(stderr, "%s:\n  Failed to create Rx Completion Queue: %s\n",
            __PRETTY_FUNCTION__, "No memory");
    return ENOMEM;
  }

  if(!_pep->listen(nLinks))
  {
    fprintf(stderr, "%s:\n  Failed to set Passive Endpoint to listening state: %s\n",
            __PRETTY_FUNCTION__, _pep->error());
    return _pep->error_num();
  }
  printf("EbLfServer is listening for %d client(s) on port %s\n",
         nLinks, port.c_str());

  return 0;
}

int EbLfServer::connect(EbLfLink** link, int tmo)
{
  CompletionQueue* txcq    = nullptr;
  uint64_t         txFlags = 0;
  Endpoint* ep = _pep->accept(tmo, _eq, txcq, txFlags, _rxcq, FI_RECV);
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
  _linkByEp[ep->endpoint()] = *link;

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
        EbLfLink* link = _linkByEp[ep];
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

int EbLfServer::shutdown(EbLfLink* link)
{
  Endpoint* ep = link->endpoint();
  if (!ep)  return -FI_ENOTCONN;

  _pep->close(ep);

  if (link)   delete link;

  return FI_SUCCESS;
}

void EbLfServer::shutdown()
{
  if (_pep)
  {
    _pep->shutdown();
    delete _pep;
    _pep = nullptr;
  }
  if (_rxcq)
  {
    _rxcq->shutdown();
    delete _rxcq;
    _rxcq = nullptr;
  }
  if (_eq)
  {
    _eq->shutdown();
    delete _eq;
    _eq = nullptr;
  }
}

int EbLfServer::pend(fi_cq_data_entry* cqEntry, int msTmo)
{
  timespec t0( {0, 0} );
  int      rc;

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
      if (t0.tv_sec)
      {
        timespec t1;
        rc = clock_gettime(CLOCK_MONOTONIC_COARSE, &t1);
        if (rc < 0)  perror("clock_gettime");

        const int64_t nsTmo = int64_t(msTmo) * 1000000l;
        int64_t       dt    = ( (t1.tv_sec  - t0.tv_sec) * 1000000000 +
                                (t1.tv_nsec - t0.tv_nsec) );
        if (dt > nsTmo)
        {
          _tmo = COMP_TMO;              // Switch to waiting after a timeout
          rc = -FI_ETIMEDOUT;
          break;
        }
      }
      else
      {
        rc = clock_gettime(CLOCK_MONOTONIC_COARSE, &t0);
        if (rc < 0)  perror("clock_gettime");
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
