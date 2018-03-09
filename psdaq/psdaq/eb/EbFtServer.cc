#include "EbFtServer.hh"

#include "Endpoint.hh"

#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;


EbFtServer::EbFtServer(const char*  addr,
                       std::string& port,
                       unsigned     nClients,
                       size_t       lclSize,
                       PeerSharing  shared) :
  EbFtBase(nClients),
  _addr(addr),
  _port(port),
  _lclSize(lclSize),
  _shared(shared == PEERS_SHARE_BUFFERS),
  _lclMem(nullptr)
{
  size_t alignment = sysconf(_SC_PAGESIZE);
  assert(lclSize & (alignment - 1) == 0);
  size_t size      = (_shared ? 1 : nClients) * lclSize;
  void*  lclMem    = nullptr;
  int    ret       = posix_memalign(&lclMem, alignment, size);
  if (ret)  perror("posix_memalign");
  assert(lclMem != nullptr);
  _base = (char*)lclMem;
}

EbFtServer::~EbFtServer()
{
  unsigned nEp = _ep.size();
  for (unsigned i = 0; i < nEp; ++i)
    if (_cqPoller && _ep[i])  _cqPoller->del(_ep[i]);
  if (_cqPoller)  delete _cqPoller;

  if (_pep)  delete _pep;

  free(_base);
}

int EbFtServer::connect(unsigned myId)
{
  int ret = 0;

  assert(_lclSize > 2 * sizeof(RemoteAddress)); // Else recode to work in a scratch buffer

  if (_base == nullptr)
  {
    fprintf(stderr, "No memory found for a region of size %zd\n", _lclSize);
    return -1;
  }

  ret = _connect();
  if (ret)
  {
    fprintf(stderr, "_connect() failed\n");
    return ret;
  }

  unsigned i        = 0;           // Not necessarily the same as the source ID
  char*    pool     = _base;
  unsigned nClients = _ep.size();
  do
  {
    ret = _exchangeIds(myId, pool, _lclSize, _ep[i], _rMr[i], _id[i]);
    if (ret)
    {
      fprintf(stderr, "_exchangeIds() failed at index %d(%d)\n", i, _id[i]);
      return ret;
    }

    ret = _syncLclMr(pool, _lclSize, _ep[i], _rMr[i], _ra[i]);
    if (ret)  return ret;

    //_rxDepth[i] = _ep[i]->fabric()->info()->rx_attr->size;
    //
    //_rOuts[i] = _postCompRecv(_ep[i], _rxDepth[i]);
    //if (_rOuts[i] < _rxDepth[i])
    //{
    //  fprintf(stderr, "Posted only %d of %d CQ buffers at index %d(%d)\n",
    //          _rOuts[i], _rxDepth[i], i, _id[i]);
    //}

    if (!_ep[i]->recv_comp_data())
    {
      fprintf(stderr, "Failed to post a CQ buffer for client index %d: %s\n",
              i, _ep[i]->error());
    }

    printf("Client %d connected\n", _id[i]);

    if (!_shared)  pool += _lclSize;
  }
  while (++i < nClients);

  _mapIds(nClients);

  return ret;
}

int EbFtServer::_connect()
{
  _pep = new PassiveEndpoint(_addr, _port.c_str());
  if (!_pep)
  {
    fprintf(stderr, "Passive Endpoint creation failed\n");
    return -1;
  }
  if (_pep->state() != EP_UP)
  {
    fprintf(stderr, "Passive Endpoint state is not UP: %s\n", _pep->error());
    return _pep->error_num();
  }

  Fabric* fab = _pep->fabric();

  printf("Server is using %s provider\n", fab->provider());

  _cqPoller = new CompletionPoller(fab, _ep.size());
  if (!_cqPoller)
  {
    fprintf(stderr, "Completion Poller creation failed\n");
    return -1;
  }

  if(!_pep->listen())
  {
    fprintf(stderr, "Failed to set passive endpoint to listening state: %s\n",
            _pep->error());
    return _pep->error_num();
  }
  printf("Listening for client(s) on port %s\n", _port.c_str());

  unsigned i        = 0;           // Not necessarily the same as the source ID
  unsigned nClients = _ep.size();
  do
  {
    _ep[i] = _pep->accept();
    if (!_ep[i])
    {
      fprintf(stderr, "Endpoint accept failed at client index %d: %s\n",
              i, _pep->error());
      return _pep->error_num();
    }

    _cqPoller->add(_ep[i]);
  }
  while (++i < nClients);

  return 0;
}

int EbFtServer::_exchangeIds(unsigned       myId,
                             char*          pool,
                             size_t         size,
                             Endpoint*      ep,
                             MemoryRegion*& mr,
                             unsigned&      id)
{
  Fabric* fab = ep->fabric();

  mr = fab->register_memory(pool, size);
  if (!mr)
  {
    fprintf(stderr, "Failed to register memory region @ %p, size %zu: %s\n",
            pool, size, fab->error());
    return fab->error_num();
  }

  if (!ep->recv_sync(pool, sizeof(id), mr))
  {
    fprintf(stderr, "Failed receiving peer's ID: %s\n", ep->error());
    return -1;
  }
  id = *(unsigned*)pool;

  *(unsigned*)pool = myId;
  if (!ep->send_sync(pool, sizeof(myId)))
  {
    fprintf(stderr, "Failed sending peer our ID: %s\n", ep->error());
    return -1;
  }

  return 0;
}

const char* EbFtServer::base() const
{
  return _base;
}

int EbFtServer::shutdown()
{
  int ret = FI_SUCCESS;

  for (unsigned i = 0; i < _ep.size(); ++i)
  {
    if (!_ep[i])  continue;

    bool cm_entry;
    struct fi_eq_cm_entry entry;
    uint32_t event;
    const int tmo = 1000;
    if (_ep[i]->event_wait(&event, &entry, &cm_entry, tmo))
    {
      if (cm_entry && (event == FI_SHUTDOWN))
      {
        _pep->close(_ep[i]);
        printf("Client %d disconnected\n", i);
      }
      else
      {
        fprintf(stderr, "Unexpected event %u - expected FI_SHUTDOWN (%u)\n", event, FI_SHUTDOWN);
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

  printf("\nEbFtServer dump:\n");
  _stats.dump();

  return ret;
}
