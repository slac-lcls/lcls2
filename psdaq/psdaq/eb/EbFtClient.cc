#include "EbFtClient.hh"

#include "Endpoint.hh"

#include <rdma/fi_rma.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>                     // For sleep()...
#include <assert.h>

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;

static const size_t scratch_size = sizeof(Fabrics::RemoteAddress);


EbFtClient::EbFtClient(StringList& peers,
                       StringList& port) :
  EbFtBase(peers.size()),
  _peers(peers),
  _port(port)
{
}

EbFtClient::~EbFtClient()
{
  unsigned nEp = _ep.size();

  for (unsigned i = 0; i < nEp; ++i)
    if (_ep[i])   delete _ep[i];
}

int EbFtClient::connect(unsigned id, unsigned tmo, size_t rmtSize)
{
  char pool[scratch_size];              // No big deal if unaligned

  for (unsigned i = 0; i < _peers.size(); ++i)
  {
    _lMr[i] = nullptr;
    int ret = _connect(_peers[i], _port[i], tmo, _ep[i]);
    if (ret)
    {
      fprintf(stderr, "_connect() failed at index %u (%s:%s)\n",
              i, _peers[i].c_str(), _port[i].c_str());
      return ret;
    }

    ret = _exchangeIds(id, pool, sizeof(pool), _ep[i], _rMr[i], _id[i]);
    if (ret)
    {
      fprintf(stderr, "_exchangeId() failed at index %u (%s:%s)\n",
              i, _peers[i].c_str(), _port[i].c_str());
      return ret;
    }

    ret = _syncRmtMr(pool, rmtSize, _ep[i], _rMr[i], _ra[i]);
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
  }

  _mapIds(_peers.size());

  return 0;
}

int EbFtClient::_connect(std::string&   peer,
                         std::string&   port,
                         unsigned       tmo,
                         Endpoint*&     ep)
{
  ep = new Endpoint(peer.c_str(), port.c_str());
  if (!ep || (ep->state() != EP_UP))
  {
    fprintf(stderr, "Failed to initialize fabrics endpoint %s:%s: %s\n",
            peer.c_str(), port.c_str(), ep->error());
    perror("new Endpoint");
    return ep ? ep->error_num() : -1;
  }

  Fabric* fab = ep->fabric();

  printf("Client is using %s provider\n", fab->provider());

  printf("Waiting for server %s on port %s\n", peer.c_str(), port.c_str());

  bool tmoEnabled = tmo != 0;
  tmo *= 10;
  while (!ep->connect() && (!tmoEnabled || --tmo))
  {
    usleep(100000);
  }
  if (tmoEnabled && (tmo == 0))
  {
    fprintf(stderr, "Timed out connecting to %s:%s: %s\n",
            peer.c_str(), port.c_str(), ep->error());
    perror("ep->connect()");
    return -1;
  }

  printf("Server %s:%s connected\n", peer.c_str(), port.c_str());

  return 0;
}

int EbFtClient::_exchangeIds(unsigned       myId,
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
    fprintf(stderr, "Failed to register memory region @ %p, sz %zu: %s\n",
            pool, size, fab->error());
    perror("fab->register_memory");
    return fab->error_num();
  }

  *(unsigned*)pool = myId;
  if (!ep->send_sync(pool, sizeof(myId)))
  {
    fprintf(stderr, "Failed sending peer our ID: %s\n", ep->error());
    return -1;
  }

  if (!ep->recv_sync(pool, sizeof(id), mr))
  {
    fprintf(stderr, "Failed receiving peer's ID: %s\n", ep->error());
    return -1;
  }
  id = *(unsigned*)pool;

  return 0;
}

int EbFtClient::shutdown()
{
  int ret = FI_SUCCESS;

  printf("\nEbFtClient dump:\n");
  _stats.dump();

  return ret;
}
