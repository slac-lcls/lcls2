#include "BatchHandler.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace Pds;
using namespace Pds::Eb;


BatchHandler::BatchHandler(std::string& port,
                           unsigned     nSources,
                           unsigned     nSlots,
                           unsigned     nBatches,
                           size_t       maxBatchSize) :
  XtcIterator(),
  _numSources  (nSources),
  _numBatches  (nBatches),
  _maxBatchSize(maxBatchSize),
  _pool        (new char[nSlots * nBatches * maxBatchSize]),
  _inlet       (port)
{
  int ret = _inlet.connect(nSources, nSlots, nBatches * maxBatchSize, _pool);
  if (ret)
  {
    fprintf(stderr, "_connect() failed");
    abort();
  }
}

BatchHandler::~BatchHandler()
{
  delete [] _pool;
}

void BatchHandler::pend()
{
  // Block until one or more of the Endpoints reports a contribution arrived
  // Find the contribution and process it

  Datagram* batch = (Datagram*)_inlet.pend();
  _process(batch);
}

int BatchHandler::process(Xtc* xtc)
{
  Datagram* contribution = (Datagram*)xtc->payload();
  return process(contribution);
}

void BatchHandler::_process(Datagram* batch)
{
  iterate(batch->xtc);
}

void BatchHandler::release(Datagram* batch)
{
  // Return the batch the BatchManager's pool
  _mgr->release(batch->seq.clock());
}
