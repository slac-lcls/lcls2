#include "MonLfClient.hh"

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


MonLfClient::MonLfClient(StringList& peers,
                         StringList& ports) :
  _peers(peers),
  _ports(ports),
  _ep(peers.size()),
  _mr(nullptr),
  _scratch(new char[scratch_size])
{
}

MonLfClient::~MonLfClient()
{
  if (_scratch)  delete [] _scratch;

  unsigned nEp = _ep.size();
  for (unsigned i = 0; i < nEp; ++i)
  {
    if (_ep[i]->fabric())  delete _ep[i]->fabric();
    if (_ep[i])            delete _ep[i];
  }
}

int MonLfClient::shutdown()
{
  int ret = FI_SUCCESS;

  return ret;
}

int MonLfClient::connect(unsigned tmo)
{
  for (unsigned i = 0; i < _peers.size(); ++i)
  {
    int ret = _connect(_peers[i], _ports[i], tmo, _ep[i]);
    if (ret)  return ret;
  }

  return 0;
}

int MonLfClient::_connect(std::string& peer,
                          std::string& port,
                          unsigned     tmo,
                          Endpoint*&   ep)
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

  _mr = fab->register_memory(_scratch, scratch_size);
  if (!_mr)
  {
    fprintf(stderr, "Failed to register memory region: %s\n", fab->error());
    return fab->error_num();
  }

  printf("Waiting for server %s:%s\n", peer.c_str(), port.c_str());

  bool tmoEnabled = tmo != 0;
  int  timeout    = tmoEnabled ? 1000 * tmo : -1; // mS
  tmo *= 10;
  do
  {
    ep = new Endpoint(fab);
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

  printf("Server[%zd] connected\n", &ep - &_ep[0]);

  return 0;
}

int MonLfClient::post(unsigned idx, uint64_t data)
{
  ssize_t   rc;
  Endpoint* ep = _ep[idx];
  if (!ep)  return -2;

  bool                  cmEntry;
  struct fi_eq_cm_entry entry;
  uint32_t              event;

  if (ep->event(&event, &entry, &cmEntry))
  {
    if (cmEntry && (event == FI_SHUTDOWN))
    {
      fprintf(stderr, "Remote side of Endpoint[%d] has shut down\n", idx);

      _ep.erase(_ep.begin() + idx);
      _ep[idx] = 0;

      return -1;
    }
  }

  if ((rc = ep->inject_data(_scratch, 0, data)) < 0)
  {
    fprintf(stderr, "inject_data failed: %s\n", ep->error());
    return rc;
  }

  return 0;
}
