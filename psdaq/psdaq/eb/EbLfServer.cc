#include "EbLfServer.hh"

#include "Endpoint.hh"

#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;


EbLfServer::EbLfServer(const char*  addr,
                       std::string& port,
                       unsigned     nClients) :
  EbLfBase(nClients),
  _addr(addr),
  _port(port),
  _pep(nullptr)
{
}

EbLfServer::~EbLfServer()
{
  if (_rxcq)  delete _rxcq;
  if (_pep)   delete _pep;
}

int EbLfServer::connect(unsigned    id,
                        void*       region,
                        size_t      size,
                        PeerSharing shared,
                        void*       ctx)
{
  int ret = _connect(id);
  if (ret)  return ret;

  for (unsigned i = 0; i < _ep.size(); ++i)
  {
    int ret = _setupMr(_ep[i], region, size, _mr[i]);
    if (ret)  return ret;

    if (shared == PER_PEER_BUFFERS)  region = (char*)region + size;

    ret = _exchangeIds(_ep[i], _mr[i], id, _id[i]);
    if (ret)  return ret;

    ret = _syncLclMr(_ep[i], _mr[i], _ra[i]);
    if (ret)  return ret;

    _rOuts[i] = _postCompRecv(_ep[i], _rxDepth, ctx);
    if (_rOuts[i] < _rxDepth)
    {
      fprintf(stderr, "Posted only %d of %d CQ buffers at index %d(%d)\n",
              _rOuts[i], _rxDepth, i, _id[i]);
    }

    printf("Client[%d] %d connected\n", i, _id[i]);
  }

  _mapIds(_ep.size());

  return 0;
}

int EbLfServer::_connect(unsigned id)
{
  _pep = new PassiveEndpoint(_addr, _port.c_str());
  if (!_pep || (_pep->state() != EP_UP))
  {
    fprintf(stderr, "Failed to create Passive Endpoint: %s\n",
            _pep ? _pep->error() : "No memory");
    return _pep ? _pep->error_num(): -FI_ENOMEM;
  }

  Fabric* fab = _pep->fabric();

  printf("Server is using '%s' provider\n", fab->provider());

  struct fi_cq_attr cq_attr = {
    .size             = 0,
    .flags            = 0,
    .format           = FI_CQ_FORMAT_DATA,
    .wait_obj         = FI_WAIT_UNSPEC,
    .signaling_vector = 0,
    .wait_cond        = FI_CQ_COND_NONE,
    .wait_set         = NULL,
  };

  _rxDepth     = fab->info()->rx_attr->size;
  cq_attr.size = _rxDepth + 1;
  _rxcq = new CompletionQueue(fab, &cq_attr, NULL);
  if (!_rxcq)
  {
    fprintf(stderr, "Failed to create RX completion queue: %s\n",
            "No memory");
    return -FI_ENOMEM;
  }

  if(!_pep->listen())
  {
    fprintf(stderr, "Failed to set passive endpoint to listening state: %s\n",
            _pep->error());
    return _pep->error_num();
  }
  printf("Listening for client(s) on port %s\n", _port.c_str());

  for (unsigned i = 0; i < _ep.size(); ++i)
  {
    int tmo = -1;
    _ep[i] = _pep->accept(tmo, nullptr, 0, _rxcq, FI_RECV);
    if (!_ep[i])
    {
      fprintf(stderr, "Failed to accept connection for Endpoint[%d]: %s\n",
              i, _ep[i]->error());
      return _ep[i]->error_num();
    }
  }

  return 0;
}

int EbLfServer::_exchangeIds(Endpoint*     ep,
                             MemoryRegion* mr,
                             unsigned      myId,
                             unsigned&     id)
{
  ssize_t  rc;
  unsigned idx = &ep - &_ep[0];
  void*    buf = mr->start();

  if ((rc = ep->recv_sync(buf, sizeof(id), mr)) < 0)
  {
    fprintf(stderr, "Failed receiving peer[%d]'s ID: %s\n",
            idx, ep->error());
    return rc;
  }
  id = *(unsigned*)buf;

  LocalAddress adx(buf, sizeof(myId), mr);
  LocalIOVec   iov(&adx, 1);
  Message      msg(&iov, 0, NULL, 0);

  *(unsigned*)buf = myId;
  if ((rc = ep->sendmsg_sync(&msg, FI_TRANSMIT_COMPLETE | FI_COMPLETION)) < 0)
  {
    fprintf(stderr, "Failed sending peer[%d] our ID: %s\n",
            idx, ep->error());
    return rc;
  }

  return 0;
}

int EbLfServer::shutdown()
{
  int ret = FI_SUCCESS;

  for (unsigned i = 0; i < _ep.size(); ++i)
  {
    if (!_ep[i])  continue;

    bool cm_entry;
    struct fi_eq_cm_entry entry;
    uint32_t event;
    const int tmo = 1000;               // mS
    if (_ep[i]->event_wait(&event, &entry, &cm_entry, tmo))
    {
      if (cm_entry && (event == FI_SHUTDOWN))
      {
        _pep->close(_ep[i]);
        printf("Client %d disconnected\n", i);
      }
      else
      {
        fprintf(stderr, "Unexpected event %u - expected FI_SHUTDOWN (%u)\n",
                event, FI_SHUTDOWN);
        ret = _ep[i]->error_num();
        _pep->close(_ep[i]);
        break;
      }
    }
    else
    {
      if (_ep[i]->error_num() != -FI_EAGAIN)
      {
        fprintf(stderr, "Waiting for event failed: %s\n", _ep[i]->error());
        ret = _ep[i]->error_num();
        _pep->close(_ep[i]);
        break;
      }
    }
  }

  printf("\nEbLfServer dump:\n");
  _stats.dump();

  return ret;
}
