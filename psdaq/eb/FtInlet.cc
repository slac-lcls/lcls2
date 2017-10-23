#include "FtInlet.hh"

#include "Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace Pds;
using namespace Pds::Fabrics;


FtInlet::FtInlet(std::string& port) :
  _port(port)
{
}

FtInlet::~FtInlet()
{
  unsigned nMr = _mr.size();
  unsigned nEp = _ep.size();

  for (unsigned i = 0; i < nMr; ++i)
    if (_mr[i])   delete _mr[i];
  //if (_mr)  delete _mr;
  for (unsigned i = 0; i < nEp; ++i)
    if (_ep[i])   delete _ep[i];
  if (_pep)  delete _pep;
}

int FtInlet::connect(char*    base,
                     unsigned nPeers,
                     size_t   size,
                     bool     shared)
{
  int ret = 0;

  _ep.resize(nPeers);
  _mr.resize(nPeers);

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

  if(!_pep->listen())
  {
    fprintf(stderr, "Failed to set passive fabrics endpoint to listening state: %s\n", _pep->error());
    return _pep->error_num();
  }
  printf("Listening\n");

  unsigned i    = 0;               // Not necessarily the same as the source ID
  char*    pool = base;
  Fabric*  fab  = _pep->fabric();
  do
  {
    _ep[i] = _pep->accept();
    if (!_ep[i])
    {
      fprintf(stderr, "Endpoint accept failed for peer %d: %s\n", i, _pep->error());
      ret = _pep->error_num();
      break;
    }

    printf("Peer %d connected: pool @ %p, size: %zd\n", i, pool, size);

    _mr[i] = fab->register_memory(pool, size);
    if (!_mr[i])
    {
      fprintf(stderr, "Failed to register memory region for peer %d @ %p, sz %zu: %s\n",
              i, pool, size, fab->error());
      return fab->error_num();
    }

    RemoteAddress ra(_mr[i]->rkey(), (uint64_t)pool, size);
    memcpy(pool, &ra, sizeof(ra));

    if (!_ep[i]->send_sync(pool, sizeof(ra), _mr[i]))
    {
      fprintf(stderr, "Sending of local memory keys to peer %d failed: %s\n", i, _ep[i]->error());
      ret = _ep[i]->error_num();
      _pep->close(_ep[i]);
      break;
    }

    if (!shared)  { pool += size; }
  }
  while (++i < nPeers);

  return ret;
}

int FtInlet::shutdown()
{
  int ret = FI_SUCCESS;

  for (unsigned i = 0; i < _ep.size(); ++i)
  {
    bool cm_entry;
    struct fi_eq_cm_entry entry;
    uint32_t event;
    if (_ep[i]->event_wait(&event, &entry, &cm_entry))
    {
      if (!cm_entry || event != FI_SHUTDOWN) {
        fprintf(stderr, "unexpected event %u - expected FI_SHUTDOWN (%u)\n", event, FI_SHUTDOWN);
        ret = _ep[i]->error_num();
        _pep->close(_ep[i]);
        continue;
      }
    }
    else
    {
      fprintf(stderr, "Wating for event failed: %s\n", _ep[i]->error());
      ret = _ep[i]->error_num();
      _pep->close(_ep[i]);
      continue;
    }

    _pep->close(_ep[i]);
  }

  return ret;
}

void* FtInlet::pend()
{
  while (1)
  {
    // Cycle through all sources and don't starve any one
    for (unsigned i = 0; i < _ep.size(); ++i) // Revisit: Awaiting select() or poll() soln
    {
      int              cqNum;
      fi_cq_data_entry cqEntry;

      if (_ep[i]->comp_wait(&cqEntry, &cqNum, 1)) // Revisit: Wait on all EPs with tmo
      {
        if ((cqNum == 1) && (cqEntry.flags & FI_REMOTE_WRITE))
        {
          // Revisit: Immediate data identifies which batch was written
          //          Better to use its address or parameters?
          //unsigned slot = (cqEntry.data >> 16) & 0xffff;
          //unsigned idx  =  cqEntry.data        & 0xffff;
          //batch = (Datagram*)&_pool[(slot * _maxBatches + idx) * _maxBatchSize];
          return cqEntry.op_context;
        }
      }
    }
  }
}
