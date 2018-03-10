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

static const size_t scratch_size = sizeof(Fabrics::RemoteAddress);


EbLfServer::EbLfServer(const char*  addr,
                       std::string& port,
                       unsigned     nClients) :
  EbLfBase(nClients),
  _addr(addr),
  _port(port)
{
}

EbLfServer::~EbLfServer()
{
  if (_pep)  delete _pep;
}

int EbLfServer::connect(unsigned id)
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

  //const unsigned rxDepth = 500;
  //if (rxDepth > fab->info()->rx_attr->size)
  //{
  //  fprintf(stderr, "rxDepth requested: %d, "
  //          "rxDepth supported: %zd\n", rxDepth, fab->info()->rx_attr->size);
  //  return -1;
  //}
  //_rxDepth = rxDepth;

  //_cqPoller = new CompletionPoller(fab, _ep.size());
  //if (!_cqPoller)
  //{
  //  fprintf(stderr, "Completion Poller creation failed\n");
  //  return -1;
  //}

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

    //if (!_cqPoller->add(_ep[i], _ep[i]->rxcq()))
    //{
    //  fprintf(stderr, "Failed to add client index %d to completion poller: %s\n",
    //          i, _cqPoller->error());
    //  return _cqPoller->error_num();
    //}

    int ret = _exchangeIds(_ep[i], id, _id[i]);
    if (ret)
    {
      fprintf(stderr, "Exchanging IDs failed at index %d\n", i);
      return ret;
    }

    printf("Client[%d] %d connected\n", i, _id[i]);
  }
  while (++i < nClients);

  _mapIds(_ep.size());

  return 0;
}

int EbLfServer::_exchangeIds(Endpoint* ep,
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

  if ((rc = ep->recv_sync(scratch, sizeof(id), mr)) < 0)
  {
    fprintf(stderr, "Failed receiving peer's ID: %s\n", ep->error());
    return rc;
  }
  id = *(unsigned*)scratch;

  *(unsigned*)scratch = myId;
  if ((rc = ep->send_sync(scratch, sizeof(myId))) < 0)
  {
    fprintf(stderr, "Failed sending peer our ID: %s\n", ep->error());
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

  printf("\nEbLfServer dump:\n");
  _stats.dump();

  return ret;
}
