#include "BatchManager.hh"

#include "xtcdata/xtc/Dgram.hh"

#include <memory>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>

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
  _maxBatchSize (sizeof(Dgram) + maxEntries * maxSize), // Note rounding up below
  _batchBuffer  (nullptr),
  _datagrams    (new const Dgram*[batchDepth * maxEntries]),
  _pool         (Batch::size(), batchDepth),
  _batches      (new Batch*[batchDepth]),
  _batch        (nullptr)
{
  if (duration & (duration - 1))
  {
    fprintf(stderr, "Batch duration (%016lx) must be a power of 2\n",
            duration);
    abort();
  }
  if (maxEntries < duration)
  {
    fprintf(stderr, "Warning: More triggers can occur in a batch duration (%lu) "
            "than for which there are batch entries (%u).\n"
            "Beware the trigger rate!\n",
            duration, maxEntries);
  }

  size_t alignment = sysconf(_SC_PAGESIZE);
  _maxBatchSize    = alignment * ((_maxBatchSize + alignment - 1) / alignment);
  size_t size      = batchDepth * _maxBatchSize;
  void*  buffer    = nullptr;
  int    ret       = posix_memalign(&buffer, alignment, size);
  if (ret)  perror("posix_memalign");
  assert(buffer != nullptr);
  _batchBuffer = (char*)buffer;
  Batch::init(_pool, _batchBuffer, batchDepth, _maxBatchSize, _datagrams, maxEntries, _batches);
}

BatchManager::~BatchManager()
{
  delete [] _batches;
  delete [] _datagrams;
  free(_batchBuffer);
}

void* BatchManager::batchRegion() const
{
  return _batchBuffer;
}

size_t BatchManager::batchRegionSize() const
{
  return _batchDepth * _maxBatchSize;
}

Batch* BatchManager::allocate(const Dgram* datagram)
{
  uint64_t pid = _startId(datagram->seq.pulseId().value());

  if (!_batch || _batch->expired(pid))
  {
    if (_batch)  post(_batch);

     // Resource wait if pool is empty
    _batch = new(&_pool) Batch(*datagram, pid);
  }

  return _batch;
}

void BatchManager::process(const Dgram* datagram)
{
  Batch* batch  = allocate(datagram);
  size_t size   = sizeof(*datagram) + datagram->xtc.sizeofPayload();
  void*  buffer = batch->allocate(size);
  batch->store(datagram);

  memcpy(buffer, datagram, size);

  ((Dgram*)buffer)->xtc.src.phy(batch->index());
}

const Batch* BatchManager::batch(unsigned index) const
{
  assert(index < _batchDepth);
  return _batches[index];
}

size_t BatchManager::maxBatchSize() const
{
  return _maxBatchSize;
}

//uint64_t BatchManager::batchId(uint64_t id) const
//{
//  return id / _duration;         // Batch number since EPOCH
//}

uint64_t BatchManager::batchId(uint64_t id) const
{
  return id >> _durationShift;          // Batch number since EPOCH
}

//uint64_t BatchManager::_startId(uint64_t id) const
//{
//  return _batchId(id) * _duration;     // Current batch ID
//}

uint64_t BatchManager::_startId(uint64_t id) const
{
  return id & _durationMask;
}

void BatchManager::dump() const
{
  printf("\nBatchManager pool:\n");
  _pool.dump();
}
