#include "EbFtServer.hh"

#include "Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;


EbFtServer::EbFtServer(std::string& port,
                       unsigned     nClients,
                       size_t       lclSize) :
  EbFtBase(nClients),
  _port(port),
  _lclSize(lclSize),
  _base(new char[nClients * lclSize])
{
}

EbFtServer::~EbFtServer()
{
  unsigned nEp = _ep.size();
  for (unsigned i = 0; i < nEp; ++i)
    if (_cqPoller && _ep[i])  _cqPoller->del(_ep[i]);
  if (_cqPoller)  delete _cqPoller;

  if (_pep)  delete _pep;

  delete [] _base;
}

int EbFtServer::connect()
{
  int ret = 0;

  assert(_lclSize > 2 * sizeof(RemoteAddress)); // Else recode to work in a scratch buffer

  if (_base == NULL)
  {
    fprintf(stderr, "No memory found for a region of size %zd\n", _lclSize);
    return -1;
  }

  _pep = new PassiveEndpoint(NULL, _port.c_str());
  if (!_pep)
  {
    fprintf(stderr, "Passive Endpoint creation failed\n");
    return -1;
  }
  if (_pep->state() != EP_UP)
  {
    fprintf(stderr, "Failed to initialize fabrics endpoint: %s\n", _pep->error());
    return _pep->error_num();
  }

  Fabric* fab = _pep->fabric();

  printf("Fabric provider is '%s'\n", fab->provider());

  _cqPoller = new CompletionPoller(fab, _ep.size());

  if(!_pep->listen())
  {
    fprintf(stderr, "Failed to set passive fabrics endpoint to listening state: %s\n", _pep->error());
    return _pep->error_num();
  }
  printf("Listening for client(s) on port %s\n", _port.c_str());

  unsigned i        = 0;           // Not necessarily the same as the source ID
  char*    pool     = _base;
  unsigned nClients = _ep.size();
  do
  {
    _ep[i] = _pep->accept();
    if (!_ep[i])
    {
      fprintf(stderr, "Endpoint accept failed for client %d: %s\n", i, _pep->error());
      ret = _pep->error_num();
      break;
    }

    printf("Client %d connected\n", i);

    _mr[i] = fab->register_memory(pool, _lclSize);
    if (!_mr[i])
    {
      fprintf(stderr, "Failed to register memory region @ %p, size %zu: %s\n",
              pool, _lclSize, fab->error());
      return fab->error_num();
    }

    _cqPoller->add(_ep[i]);

    ret = _syncLclMr(pool, _lclSize, _ep[i], _mr[i]);
    if (ret)  break;

    _ep[i]->recv_comp_data();

    pool += _lclSize;
  }
  while (++i < nClients);

  return ret;
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
    if (_ep[i]->event_wait(&event, &entry, &cm_entry))
    {
      if (cm_entry && (event == FI_SHUTDOWN))
      {
        _pep->close(_ep[i]);
        printf("Client %d disconnected!\n", i);
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
      fprintf(stderr, "Wating for event failed: %s\n", _ep[i]->error());
      ret = _ep[i]->error_num();
      _pep->close(_ep[i]);
      break;
    }
  }

  return ret;
}
