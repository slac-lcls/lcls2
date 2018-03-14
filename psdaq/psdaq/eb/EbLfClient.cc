#include "EbLfClient.hh"

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


EbLfClient::EbLfClient(StringList& peers,
                       StringList& port) :
  EbLfBase(peers.size()),
  _peers(peers),
  _port(port)
{
}

EbLfClient::~EbLfClient()
{
  unsigned nEp = _ep.size();

  for (unsigned i = 0; i < nEp; ++i)
    if (_ep[i])   delete _ep[i];
}

int EbLfClient::connect(unsigned id, unsigned tmo)
{
  for (unsigned i = 0; i < _peers.size(); ++i)
  {
    int ret = _connect(_peers[i], _port[i], tmo, _ep[i]);
    if (ret)
    {
      fprintf(stderr, "_connect() failed at index %u (%s:%s)\n",
              i, _peers[i].c_str(), _port[i].c_str());
      return ret;
    }

    ret = _exchangeIds(_ep[i], id, _id[i]);
    if (ret)
    {
      fprintf(stderr, "Exchanging IDs failed at index %d\n", i);
      return ret;
    }

    printf("Server[%d] %d connected\n", i, _id[i]);
  }

  _mapIds(_peers.size());

  return 0;
}

int EbLfClient::_connect(std::string& peer,
                         std::string& port,
                         unsigned     tmo,
                         Endpoint*&   ep)
{
  ep = new Endpoint(peer.c_str(), port.c_str());
  if (!ep || (ep->state() != EP_UP))
  {
    fprintf(stderr, "Failed to initialize fabrics endpoint %s:%s: %s\n",
            peer.c_str(), port.c_str(), ep ? ep->error() : "");
    if (!ep)  perror("new Endpoint");
    return ep ? ep->error_num() : errno;
  }

  Fabric* fab = ep->fabric();

  printf("Client is using %s provider\n", fab->provider());

  //if (!_rxDepth)
  //{
  //  const unsigned rxDepth = 500;
  //  if (rxDepth > fab->info()->rx_attr->size)
  //  {
  //    fprintf(stderr, "rxDepth requested: %d, "
  //            "rxDepth supported: %zd\n", rxDepth, fab->info()->rx_attr->size);
  //    return -1;
  //  }
  //  _rxDepth = rxDepth;

  //if (!_cqPoller)
  //{
  //  _cqPoller = new CompletionPoller(fab, _ep.size());
  //  if (!_cqPoller)
  //  {
  //    fprintf(stderr, "Completion Poller creation failed\n");
  //    return -1;
  //  }
  //}
  //
  //if (!_cqPoller->add(ep, ep->rxcq()))
  //{
  //  fprintf(stderr, "Failed to add Endpoing to completion poller: %s\n",
  //          _cqPoller->error());
  //  return _cqPoller->error_num();
  //}

  printf("Waiting for server %s on port %s\n", peer.c_str(), port.c_str());

  int  ret;
  bool tmoEnabled = tmo != 0;
  int  timeout    = tmoEnabled ? 1000 * tmo : -1;
  tmo *= 10;
  while (!(ret = ep->connect(timeout)) && tmoEnabled && --tmo)
  {
    if (ep->error_num() == -FI_ENODATA)  break;
    usleep(100000);
  }
  if (!ret || (tmoEnabled && (tmo == 0)))
  {
    const char* msg = tmoEnabled ? "Timed out connecting" : "Failed to connect";
    fprintf(stderr, "%s to %s:%s: %s\n", msg,
            peer.c_str(), port.c_str(), ep->error());
    return ep->error_num();
  }

  printf("Server %s:%s connected\n", peer.c_str(), port.c_str());

  return 0;
}

int EbLfClient::_exchangeIds(Endpoint* ep,
                             unsigned  myId,
                             unsigned& id)
{
  ssize_t rc;
  char    scratch[scratch_size];           // No big deal if unaligned
  Fabric* fab = ep->fabric();

  MemoryRegion* mr = fab->register_memory(scratch, sizeof(scratch));
  if (!mr)
  {
    fprintf(stderr, "Failed to register memory region @ %p, size %zu: %s\n",
            scratch, sizeof(scratch), fab->error());
    return fab->error_num();
  }

  *(unsigned*)scratch = myId;
  if ((rc = ep->send_sync(scratch, sizeof(myId))) < 0)
  {
    fprintf(stderr, "Failed sending peer our ID: %s\n", ep->error());
    return rc;
  }

  if ((rc = ep->recv_sync(scratch, sizeof(id), mr)) < 0)
  {
    fprintf(stderr, "Failed receiving peer's ID: %s\n", ep->error());
    return rc;
  }
  id = *(unsigned*)scratch;

  return 0;
}

int EbLfClient::shutdown()
{
  int ret = FI_SUCCESS;

  printf("\nEbLfClient dump:\n");
  _stats.dump();

  return ret;
}
