#include "EbFtClient.hh"

#include "Endpoint.hh"

#include <rdma/fi_rma.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>                     // For sleep()...

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;

static const size_t scratch_size = sizeof(Fabrics::RemoteAddress);


EbFtClient::EbFtClient(StringList& peers,
                       StringList& port,
                       size_t      rmtSize) :
  EbFtBase(peers.size()),
  _peers(peers),
  _port(port),
  _lclSize(scratch_size),
  _rmtSize(rmtSize),
  _base(new char[peers.size() * scratch_size])
{
}

EbFtClient::~EbFtClient()
{
  unsigned nEp = _ep.size();

  for (unsigned i = 0; i < nEp; ++i)
    if (_ep[i])   delete _ep[i];

  delete[] _base;
}

int EbFtClient::connect(unsigned id, unsigned tmo)
{
  if (_base == NULL)
  {
    fprintf(stderr, "No memory available for a region of size %zd\n", _lclSize);
    return -1;
  }

  char* pool = _base;
  for (unsigned i = 0; i < _peers.size(); ++i)
  {
    int ret = _connect(id, _peers[i], _port[i], tmo, pool, _ep[i], _mr[i], _id[i]);
    if (ret)
    {
      fprintf(stderr, "_connect() failed at index %u (%s:%s)\n",
              i, _peers[i].c_str(), _port[i].c_str());
      return ret;
    }

    ret = _syncRmtMr(pool, _rmtSize, _ep[i], _mr[i], _ra[i]);
    if (ret)  return ret;

    _ep[i]->recv_comp_data();

    pool += scratch_size;
  }

  _mapIds(_peers.size());

  return 0;
}

int EbFtClient::_connect(unsigned       myId,
                         std::string&   peer,
                         std::string&   port,
                         unsigned       tmo,
                         char*          pool,
                         Endpoint*&     ep,
                         MemoryRegion*& mr,
                         unsigned&      id)
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

  mr = fab->register_memory(pool, _lclSize);
  if (!mr)
  {
    fprintf(stderr, "Failed to register memory region @ %p, sz %zu: %s\n",
            pool, _lclSize, fab->error());
    perror("fab->register_memory");
    return fab->error_num();
  }

  printf("Waiting for server %s on port %s\n", peer.c_str(), port.c_str());

  bool tmoEnabled = tmo != 0;
  while (!ep->connect() && (!tmoEnabled || --tmo))
  {
    sleep (1);
  }
  if (tmoEnabled && (tmo == 0))
  {
    fprintf(stderr, "Timed out connecting to %s:%s: %s\n",
            peer.c_str(), port.c_str(), ep->error());
    perror("ep->connect()");
    return -1;
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

  printf("Server %d (%s:%s) connected\n", id, peer.c_str(), port.c_str());

  return 0;
}

int EbFtClient::shutdown()
{
  int ret = FI_SUCCESS;

  for (unsigned i = 0; i < _ep.size(); ++i)
  {
    if (!_ep[i])  continue;

    _ep[i]->shutdown();
  }

  return ret;
}
