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


EbLfClient::EbLfClient(StringList& peers,
                       StringList& ports) :
  EbLfBase(peers.size()),
  _peers(peers),
  _ports(ports)
{
}

EbLfClient::~EbLfClient()
{
  unsigned nEp = _ep.size();
  for (unsigned i = 0; i < nEp; ++i)
  {
    if (_ep[i]->fabric())  delete _ep[i]->fabric();
    if (_ep[i])            delete _ep[i];
  }

  for (unsigned i = 0; i < nEp; ++i)
  {
    if (_txcq[i])  delete _txcq[i];
  }
}

int EbLfClient::connect(unsigned    id,
                        unsigned    tmo,
                        void*       region,
                        size_t      size,
                        PeerSharing shared,
                        void*       ctx)
{
  for (unsigned i = 0; i < _peers.size(); ++i)
  {
    int ret = _connect(_peers[i], _ports[i], tmo, _ep[i], _txcq[i]);
    if (ret)  return ret;

    ret = _setupMr(_ep[i], region, size, _mr[i]);
    if (ret)  return ret;

    if (shared == PER_PEER_BUFFERS)  region = (char*)region + size;

    ret = _exchangeIds(_ep[i], _mr[i], id, _id[i]);
    if (ret)  return ret;

    ret = _syncRmtMr(_ep[i], _mr[i], _ra[i], size);
    if (ret)  return ret;

    _rOuts[i] = _postCompRecv(_ep[i], _rxDepth, ctx);
    if (_rOuts[i] < _rxDepth)
    {
      fprintf(stderr, "Posted only %d of %d CQ buffers at index %d(%d)\n",
              _rOuts[i], _rxDepth, i, _id[i]);
    }

    printf("Server[%d] %d connected\n", i, _id[i]);
  }

  _mapIds(_peers.size());

  return 0;
}

int EbLfClient::_connect(std::string&      peer,
                         std::string&      port,
                         unsigned          tmo,
                         Endpoint*&        ep,
                         CompletionQueue*& txcq)
{
  unsigned idx = &peer - &_peers[0];

  Fabric* fab = new Fabric(peer.c_str(), port.c_str());
  if (!fab || !fab->up())
  {
    fprintf(stderr, "Failed to create Fabric[%d]: %s\n",
            idx, fab ? fab->error() : "No memory");
    return fab ? fab->error_num() : -FI_ENOMEM;
  }

  printf("Client[%d] is using '%s' provider\n", idx, fab->provider());

  struct fi_cq_attr cq_attr = {
    .size             = 0,
    .flags            = 0,
    .format           = FI_CQ_FORMAT_DATA,
    .wait_obj         = FI_WAIT_UNSPEC,
    .signaling_vector = 0,
    .wait_cond        = FI_CQ_COND_NONE,
    .wait_set         = NULL,
  };

  cq_attr.size = fab->info()->tx_attr->size + 1;
  txcq = new CompletionQueue(fab, &cq_attr, NULL);
  if (!txcq)
  {
    fprintf(stderr, "Failed to create TX completion queue[%d]: %s\n",
            idx, "No memory");
    return -FI_ENOMEM;
  }

  printf("Waiting for server %s:%s\n", peer.c_str(), port.c_str());

  bool tmoEnabled = tmo != 0;
  int  timeout    = tmoEnabled ? 1000 * tmo : -1; // mS
  tmo *= 10;
  do
  {
    ep = new Endpoint(fab, txcq, nullptr);
    if (!ep || (ep->state() != EP_UP))
    {
      fprintf(stderr, "Failed to initialize Endpoint[%d]: %s\n",
              idx, ep ? ep->error() : "No memory");
      return ep ? ep->error_num() : -FI_ENOMEM;
    }

    if (ep->connect(timeout, FI_TRANSMIT | FI_SELECTIVE_COMPLETION, 0))  break;
    if (ep->error_num() == -FI_ENODATA)  break; // connect() timed out

    delete ep;                      // Can't try to connect on an EP a 2nd time

    usleep(100000);
  }
  while (tmoEnabled && --tmo);
  if ((ep->error_num() != FI_SUCCESS) || (tmoEnabled && (tmo == 0)))
  {
    const char* msg = tmoEnabled ? "Timed out connecting" : "Failed to connect";
    fprintf(stderr, "%s to %s:%s: %s\n", msg,
            peer.c_str(), port.c_str(), ep->error());
    return ep->error_num();
  }

  return 0;
}

int EbLfClient::_exchangeIds(Endpoint*     ep,
                             MemoryRegion* mr,
                             unsigned      myId,
                             unsigned&     id)
{
  ssize_t      rc;
  unsigned     idx = &ep - &_ep[0];
  void*        buf = mr->start();

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

  if ((rc = ep->recv_sync(buf, sizeof(id), mr)) < 0)
  {
    fprintf(stderr, "Failed receiving peer[%d]'s ID: %s\n",
            idx, ep->error());
    return rc;
  }
  id = *(unsigned*)buf;

  return 0;
}

int EbLfClient::shutdown()
{
  int ret = FI_SUCCESS;

  printf("\nEbLfClient dump:\n");
  _stats.dump();

  return ret;
}
