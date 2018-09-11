#include "BatchManager.hh"

#include "psdaq/eb/utilities.hh"

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
  _durationShift(__builtin_ctzl(duration)),
  _durationMask (~(duration - 1) & ((1UL << PulseId::NumPulseIdBits) - 1)),
  _batchDepth   (batchDepth),
  _maxEntries   (maxEntries),
  _maxBatchSize (roundUpSize(sizeof(Dgram) + maxEntries * maxSize)),
  _batchBuffer  ((char*)allocRegion(batchDepth * _maxBatchSize)),
  _batchFreeList(batchDepth, nullptr),
  _batches      (batchDepth),
  _appPrms      (new std::atomic<uintptr_t>[batchDepth * maxEntries]),
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

  char*                   buffer  = _batchBuffer;
  std::atomic<uintptr_t>* appPrms = _appPrms;
  for (unsigned i = 0; i < batchDepth; ++i)
  {
    _batches[i]._fixup(i, buffer, appPrms);
    _batchFreeList.push(&_batches[i]);
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
  Batch* batch = _batch;

  if (!batch || batch->expired(idg->seq.pulseId().value(), _durationMask))
  {
    if (batch)  post(batch);

    const auto tmo(std::chrono::milliseconds(5000));
    batch  = _batchFreeList.pop(tmo);   // Waits when pool is empty
    if (batch)  batch->initialize(idg);
    else printf("Batch pop timeout\n");
    _batch = batch;
  }

  return batch;
}

void BatchManager::dump() const
{
  printf("\nBatchManager batch free list count: %zd / %zd\n\n",
         _batchFreeList.count(), _batchFreeList.size());
}
