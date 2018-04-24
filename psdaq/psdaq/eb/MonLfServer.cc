#include "MonLfServer.hh"

#include "Endpoint.hh"

#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;


MonLfServer::MonLfServer(const char*  addr,
                         std::string& port) :
  _addr(addr),
  _port(port),
  _pep(nullptr),
  _mr(nullptr),
  _rxcq(nullptr),
  _scratch(new char[scratch_size]),
  _running(true),
  _listener(nullptr)
{
}

MonLfServer::~MonLfServer()
{
  if (_listener)  delete    _listener;
  if (_scratch)   delete [] _scratch;
  if (_rxcq)      delete    _rxcq;
  if (_pep)       delete    _pep;
}

int MonLfServer::shutdown()
{
  int ret = FI_SUCCESS;

  _running = false;

  if (_listener)  _listener->join();

  return ret;
}

int MonLfServer::connect()
{
  _pep = new PassiveEndpoint(_addr, _port.c_str());
  if (!_pep || (_pep->state() != EP_UP))
  {
    fprintf(stderr, "Failed to create Passive Endpoint: %s\n",
            _pep ? _pep->error() : "No memory");
    return _pep ? _pep->error_num(): -FI_ENOMEM;
  }

  Fabric* fab = _pep->fabric();

  printf("Server is using '%s' provider\n", fab->provider());

  _mr = fab->register_memory(_scratch, scratch_size);
  if (!_mr)
  {
    fprintf(stderr, "Failed to register memory region: %s\n", fab->error());
    return fab->error_num();
  }

  struct fi_cq_attr cq_attr = {
    .size             = 0,
    .flags            = 0,
    .format           = FI_CQ_FORMAT_DATA,
    .wait_obj         = FI_WAIT_UNSPEC,
    .signaling_vector = 0,
    .wait_cond        = FI_CQ_COND_NONE,
    .wait_set         = NULL,
  };

  _rxDepth     = fab->info()->rx_attr->size;
  cq_attr.size = _rxDepth + 1;
  _rxcq = new CompletionQueue(fab, &cq_attr, NULL);
  if (!_rxcq)
  {
    fprintf(stderr, "Failed to create RX completion queue: %s\n",
            "No memory");
    return -FI_ENOMEM;
  }

  if(!_pep->listen())
  {
    fprintf(stderr, "Failed to set passive endpoint to listening state: %s\n",
            _pep->error());
    return _pep->error_num();
  }
  printf("Listening for client(s) on port %s\n", _port.c_str());

  _listener = new std::thread([&] { _listen(); });

  return 0;
}

int MonLfServer::postCompRecv(unsigned idx, void* ctx)
{
  if (--_rOuts[idx] <= 1)
  {
    unsigned count = _rxDepth - _rOuts[idx];
    _rOuts[idx] += _postCompRecv(_ep[idx], count, ctx);
    if (_rOuts[idx] < _rxDepth)
    {
      fprintf(stderr, "Failed to post all %d receives for index %d: %d\n",
              count, idx, _rOuts[idx]);
    }
  }

  return _rOuts[idx];
}

int MonLfServer::_postCompRecv(Endpoint* ep, unsigned count, void* ctx)
{
  unsigned i;

  for (i = 0; i < count; ++i)
  {
    ssize_t rc;
    if ((rc = ep->recv_comp_data(ctx)) < 0)
    {
      if (rc != -FI_EAGAIN)
        fprintf(stderr, "Failed to post a CQ buffer: %s\n", ep->error());
      break;
    }
  }

  return i;
}

void MonLfServer::_listen()
{
  printf ("Listener starting\n");

  while (_running)
  {
    int tmo = 1000;                      // ms
    Endpoint *ep = _pep->accept(tmo, nullptr, 0, _rxcq, FI_RECV);
    if (ep)
    {
      size_t idx = _ep.size();
      printf("Client[%zd] connected\n", idx);
      int rOuts = _postCompRecv(ep, _rxDepth);
      if (rOuts < _rxDepth)
      {
        fprintf(stderr, "Posted only %d of %d CQ buffers at index %zd\n",
                rOuts, _rxDepth, idx);
      }

      // Exchange IDs here?
      // Maybe put ep and rOuts in an unordered_map here?

      _ep.push_back(ep);
      _rOuts.push_back(rOuts);
    }

    for (unsigned i = 0; i < _ep.size(); ++i)
    {
      bool                  cmEntry;
      struct fi_eq_cm_entry entry;
      uint32_t              event;

      if (_ep[i] && _ep[i]->event(&event, &entry, &cmEntry))
      {
        if (cmEntry && (event == FI_SHUTDOWN))
        {
          _pep->close(_ep[i]);
          _ep.erase(_ep.begin() + i);
          _ep[i]=0;
          printf("Client[%d] disconnected\n", i);
        }
      }
    }
  }

  printf("Listener exiting\n");
}

int MonLfServer::poll(uint64_t* data)
{
  fi_cq_data_entry cqEntry;
  const int        maxCnt = 1;
  ssize_t rc = _rxcq->comp(&cqEntry, maxCnt);
  if (rc == maxCnt)
  {
    const uint64_t flags = FI_MSG | FI_RECV | FI_REMOTE_CQ_DATA;
    if ((cqEntry.flags & flags) == flags)
    {
      *data = cqEntry.data;
    }
  }
  else if (rc == -FI_EAGAIN)
  {
    rc = 0;
  }

  return rc;
}
