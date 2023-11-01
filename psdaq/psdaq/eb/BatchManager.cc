#include "BatchManager.hh"

#include "utilities.hh"

#include <new>
#include <memory>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>                     // For memset()

using namespace Pds;
using namespace Pds::Eb;


BatchManager::BatchManager() :
  _regSize     (0),
  _region      (nullptr),
  _maxEntrySize(0),
  _maxBatchSize(0)
{
}

BatchManager::~BatchManager()
{
  if (_region)  free(_region);
}

int BatchManager::initialize(size_t maxEntrySize, unsigned maxEntries, unsigned numBatches)
{
  if (maxEntries & (maxEntries - 1))
  {
    fprintf(stderr, "%s: maxEntries must be a power of 2, got %u = %08x\n",
            __PRETTY_FUNCTION__, maxEntries, maxEntries);
    return 1;
  }

  // Reallocate the region if the required size has changed
  auto regSize = numBatches * maxEntries * maxEntrySize;
  if (regSize != _regSize)
  {
    if (_region)  free(_region);

    _region = static_cast<char*>(allocRegion(regSize));
    if (!_region)
    {
      fprintf(stderr, "%s: No memory found for a region of size %zd\n",
              __PRETTY_FUNCTION__, regSize);
      return ENOMEM;
    }

    // Save the allocated size, which may be more than the required size
    _regSize = regSize;
  }

  _mask         = ~uint64_t(maxEntries - 1);
  _maxEntrySize = maxEntrySize;
  _maxBatchSize = maxEntries * maxEntrySize;

  return 0;
}

void BatchManager::shutdown()
{
  if (_region)  memset(_region, 0, _regSize);
}

void BatchManager::dump() const
{
  printf("\nBatchManager dump:\n");
  printf("  Region base %p  size %zd  maxBatchSize %zd  maxEntrySize %zd\n",
         _region, _regSize, _maxBatchSize, _maxEntrySize);
}
