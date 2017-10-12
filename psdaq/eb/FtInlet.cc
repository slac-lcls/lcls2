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
  if (_mr)   delete _mr;
  if (_ep)   delete _ep;
  if (_pep)  delete _pep;
}

int FtInlet::connect(unsigned nPeers,
                     unsigned nSlots,
                     size_t   size,
                     char*    base)
{
  int ret;

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
  uint64_t pool = base - (char*)0;
  Fabric*  fab  = _pep->fabric();
  do
  {
    _mr[i] = fab->register_memory(pool, size);
    if (!_mr[i])
    {
      fprintf(stderr, "Failed to register memory region for peer %d @ %p, sz %zu: %s\n",
              i, pool, size, fab->error());
      ret = fab->error_num();
      break;
    }

    _ep[i] = _pep->accept();
    if (!_ep[i])
    {
      fprintf(stderr, "Endpoint accept failed for peer %d: %s\n", i, _pep->error());
      ret = _pep->error_num();
      break;
    }

    printf("Peer %d connected!\n", i);

    RemoteAddress ra(mr->rkey(), pool, size);
    memcpy(pool, &ra, sizeof(ra));

    if (!_ep[i]->send_sync(pool, sizeof(ra), mr))
    {
      fprintf(stderr, "Sending of local memory keys to peer %d failed: %s\n", i, ep->error());
      ret = _ep[i]->error_num();
      _pep->close(_ep[i]);
      break;
    }

    pool += size;
  }
  while (++i < nPeers);

  return ret;
}

void FtInlet::shutdown()
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
        fprintf(stderr, "unexpected event %u - expected FI_SHUTDOWN (%u)", event, FI_SHUTDOWN);
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

uint64_t BatchHandler::pend()
{
  while (1)
  {
    // Cycle through all sources and don't starve any one
    for (unsigned i = 0; i < _ep.size(); ++i) // Revisit: Awaiting select() or poll() soln
    {
      unsigned         cqNum;
      fi_cq_data_entry cqEntry;

      if (_ep[i]->comp(&cqEntry, &cqNum, 1)) // Revisit: Wait on all EPs with tmo
      {
        if ((cqNum == 1) && (cqEntry.flags & FI_REMOTE_WRITE))
        {
          // Revisit: Immediate data identifies which batch was written
          //          Better to use its address or parameters?
          //unsigned slot = (cqEntry.data >> 16) & 0xffff;
          //unsigned idx  =  cqEntry.data        & 0xffff;
          //batch = (Datagram*)&_pool[(slot * _numBatches + idx) * _maxBatchSize];
          return cqEntry.data;
        }
      }
    }
  }
}
