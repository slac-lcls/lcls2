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


BatchManager::BatchManager(size_t maxSize) :
  _maxSize      (maxSize),
  _maxBatchSize (roundUpSize(MAX_ENTRIES * maxSize)),
  _region       (static_cast<char*>(allocRegion(batchRegionSize()))),
  _batchFreelist(MAX_BATCHES),
  _appPrms      (new AppPrm[MAX_BATCHES * MAX_ENTRIES]),
  _batch        (nullptr)
{
  if (MAX_ENTRIES < BATCH_DURATION)
  {
    fprintf(stderr, "%s: Warning: More triggers can occur in a batch duration (%lu) "
            "than for which there are batch entries (%u).\n"
            "Beware the trigger rate!\n",
            __func__, BATCH_DURATION, MAX_ENTRIES);
  }
  if (maxSize % sizeof(uint64_t) != 0)
  {
    fprintf(stderr, "%s: Warning: Make max Dgram buffer size (%zd) divisible "
            "by %zd to avoid alignment issues",
            __func__, maxSize, sizeof(uint64_t));
  }
  if (_region == nullptr)
  {
    fprintf(stderr, "%s: No memory found for a region of size %zd\n",
            __func__, batchRegionSize());
    abort();
  }
  if (_appPrms == nullptr)
  {
    fprintf(stderr, "%s: No memory found for %d application parameters\n",
            __func__, MAX_BATCHES * MAX_ENTRIES);
    abort();
  }

  char*   buffer  = _region;
  AppPrm* appPrms = _appPrms;
  for (unsigned i = 0; i < MAX_BATCHES; ++i)
  {
    new(_batchFreelist[i]) Batch(buffer, appPrms);
    buffer  += _maxBatchSize;
    appPrms += MAX_ENTRIES;
  }
}

BatchManager::~BatchManager()
{
  if (_appPrms)  delete [] _appPrms;
  free(_region);
}

void BatchManager::shutdown()
{
  _batchFreelist.clear();
  _batch = nullptr;

  memset(_region, 0, batchRegionSize() * sizeof(*_region));
  memset(_appPrms, 0, MAX_BATCHES * MAX_ENTRIES * sizeof(*_appPrms));
}

void BatchManager::dump() const
{
  printf("\nBatchManager batch freelist:\n");
  _batchFreelist.dump();
}
