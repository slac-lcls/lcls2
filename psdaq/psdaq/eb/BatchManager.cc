#include "BatchManager.hh"

#include "utilities.hh"

#include "xtcdata/xtc/Dgram.hh"

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
  _durationMask (~(duration - 1) & ((1UL << PulseId::NumPulseIdBits) - 1)),
  _batchDepth   (batchDepth),
  _maxEntries   (maxEntries),
  _maxBatchSize (roundUpSize(sizeof(Dgram) + maxEntries * maxSize)),
  _batchBuffer  ((char*)allocRegion(batchDepth * _maxBatchSize)),
  _batchFreelist(batchDepth),
  _appPrms      (new AppPrm[batchDepth * maxEntries]),
  _batch        (nullptr)
{
  if (duration & (duration - 1))
  {
    fprintf(stderr, "%s: Batch duration (%016lx) must be a power of 2\n",
            __func__, duration);
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
    _batchFreelist[i]._fixup(i, buffer, appPrms);
    buffer  += _maxBatchSize;
    appPrms += maxEntries;
  }
}

BatchManager::~BatchManager()
{
  if (_appPrms)  delete [] _appPrms;
  free(_batchBuffer);
}

Batch* BatchManager::allocate(const Dgram* idg)
{
  uint64_t pid   = idg->seq.pulseId().value();
  Batch*   batch = _batch;

  if (!batch || batch->expired(pid, _durationMask))
  {
    if (batch)  post(batch);

    const auto tmo(std::chrono::milliseconds(5000));
    batch  = _batchFreelist.allocate(batchId(pid), tmo);
    if (batch)  batch->initialize(idg);
    //else printf("Batch allocation timeout for ID 0x%014lx\n", pid);
    _batch = batch;
  }

  return batch;
}

void BatchManager::dump() const
{
  printf("\nBatchManager batch freelist:\n");
  _batchFreelist.dump();
}
