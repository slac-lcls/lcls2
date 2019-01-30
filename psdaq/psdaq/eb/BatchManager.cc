#include "BatchManager.hh"

#include "utilities.hh"

#include "xtcdata/xtc/Dgram.hh"

#include <new>
#include <memory>
#include <stdlib.h>
#include <unistd.h>

using namespace XtcData;
using namespace Pds;
using namespace Pds::Eb;

BatchManager::BatchManager(uint64_t duration,
                           unsigned batchDepth,
                           unsigned maxEntries,
                           size_t   maxSize) :
  _duration     (duration),
  _batchDepth   (batchDepth),
  _maxEntries   (maxEntries),
  _maxSize      (maxSize),
  _maxBatchSize (roundUpSize(maxEntries * maxSize)),
  _batchBuffer  ((char*)allocRegion(batchDepth * _maxBatchSize)),
  _batchFreelist(batchDepth),
  _appPrms      (new AppPrm[batchDepth * maxEntries]),
  _batch        (nullptr)
{
  if (duration & (duration - 1))
  {
    fprintf(stderr, "%s: Batch duration (0x%016lx) must be a power of 2\n",
            __func__, duration);
    abort();
  }
  if (batchDepth & (batchDepth - 1))
  {
    fprintf(stderr, "%s: Batch depth (0x%08x) must be a power of 2\n",
            __func__, batchDepth);
    abort();
  }
  if (maxEntries < duration)
  {
    fprintf(stderr, "%s: Warning: More triggers can occur in a batch duration (%lu) "
            "than for which there are batch entries (%u).\n"
            "Beware the trigger rate!\n",
            __func__, duration, maxEntries);
  }
  if (_batchBuffer == nullptr)
  {
    fprintf(stderr, "%s: No memory found for a region of size %zd\n",
            __func__, batchDepth * _maxBatchSize);
    abort();
  }
  if (_appPrms == nullptr)
  {
    fprintf(stderr, "%s: No memory found for %d application parameters\n",
            __func__, batchDepth * maxEntries);
    abort();
  }

  char*   buffer  = _batchBuffer;
  AppPrm* appPrms = _appPrms;
  for (unsigned i = 0; i < batchDepth; ++i)
  {
    new(_batchFreelist[i]) Batch(i, buffer, appPrms);
    buffer  += _maxBatchSize;
    appPrms += maxEntries;
  }
}

BatchManager::~BatchManager()
{
  if (_appPrms)  delete [] _appPrms;
  free(_batchBuffer);
}

Batch* BatchManager::locate(uint64_t pid)
{
  Batch* batch = _batch;

  if (!batch || batch->expired(pid, ~(_duration - 1)))
  {
    if (batch)  post(batch);

    const auto tmo(std::chrono::milliseconds(5000));
    uint64_t   key(batchId(pid));
    batch  = _batchFreelist.allocate(key, tmo);
    _batch = batch;
  }

  return batch;
}

void BatchManager::dump() const
{
  printf("\nBatchManager batch freelist:\n");
  _batchFreelist.dump();
}
