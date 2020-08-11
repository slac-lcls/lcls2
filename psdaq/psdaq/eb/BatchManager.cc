#include "BatchManager.hh"

#include "utilities.hh"

#include "xtcdata/xtc/Dgram.hh"

#include <new>
#include <memory>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>                     // For memset()

using namespace XtcData;
using namespace Pds;
using namespace Pds::Eb;

static_assert((MAX_BATCHES - 1) <= ImmData::MaxIdx, "MAX_BATCHES exceeds available range");


BatchManager::BatchManager() :
  _maxBatchSize(0),
  _regSize     (0),
  _region      (nullptr),
  _appPrms     (MAX_BATCHES * MAX_ENTRIES),
  _numAllocs   (0),
  _numFrees    (0),
  _nAllocs     (0),
  _nFrees      (0),
  _waiting     (0),
  _terminate   (false)
{
}

BatchManager::~BatchManager()
{
  if (_region)  free(_region);
}

int BatchManager::initialize(size_t maxEntrySize, bool batching)
{
  if (MAX_ENTRIES < BATCH_DURATION)
  {
    fprintf(stderr, "%s: Warning: More triggers can occur in a batch duration (%lu) "
            "than for which there are batch entries (%u).\n"
            "Beware the trigger rate!\n",
            __PRETTY_FUNCTION__, BATCH_DURATION, MAX_ENTRIES);
  }
  if (maxEntrySize % sizeof(uint64_t) != 0)
  {
    fprintf(stderr, "%s: Warning: Make max EbDgram buffer size (%zd) divisible "
            "by %zd to avoid alignment issues\n",
            __PRETTY_FUNCTION__, maxEntrySize, sizeof(uint64_t));
  }

  // Allocate the region, and reallocate if the required size is larger
  _maxBatchSize = MAX_ENTRIES * maxEntrySize;
  //printf("*** BM::init: region %p, regSize %zu, regSizeGuess %zu\n",
  //       _region, _regSize, batchRegionSize());
  if (batchRegionSize() > _regSize)
  {
    //printf("*** BM::init: batchRegionSize() [%zu] > _regSize [%zu]: (re)allocating\n", batchRegionSize(), _regSize);

    if (_region)  free(_region);

    _region = static_cast<char*>(allocRegion(batchRegionSize()));
    if (!_region)
    {
      fprintf(stderr, "%s: No memory found for a region of size %zd\n",
              __PRETTY_FUNCTION__, batchRegionSize());
      return ENOMEM;
    }

    // Save the allocated size, which may be more than the required size
    _regSize = batchRegionSize();
  }

  _lastFreed    = 0;
  _batch.initialize(maxEntrySize);
  _batching     = batching;
  _numAllocs    = 0;
  _numFrees     = 0;
  _nAllocs      = 0;
  _nFrees       = 0;
  _waiting      = 0;
  _terminate    = false;

  return 0;
}

void BatchManager::shutdown()
{
  if (_region)  memset(_region, 0, batchRegionSize());
  _appPrms.clear();
}

void BatchManager::dump() const
{
  printf("\nBatchManager dump:\n");
  printf("  Region base %p  size %zd  maxBatchSize %zd\n",
         batchRegion(), batchRegionSize(), maxBatchSize());
  printf("  Number of allocs  %lu  frees %lu  diff %ld  waiting %c\n",
         batchAllocCnt(), batchFreeCnt(), batchAllocCnt() - batchFreeCnt(),
         batchWaiting() ? 'T' : 'F');
}
