#include "BatchManager.hh"

#include "xtcdata/xtc/Dgram.hh"

#include <string.h>
#include <assert.h>

using namespace XtcData;
using namespace Pds;
using namespace Pds::Eb;

BatchManager::BatchManager(Src      src,
                           uint64_t duration, // = ~((1 << N) - 1) = 128 uS?
                           unsigned batchDepth,
                           unsigned maxEntries,
                           size_t   maxSize) :
  _src          (src),
  _duration     (duration),
  _durationShift(__builtin_ctzl(duration)),
  _durationMask (~((1 << __builtin_ctzl(duration)) - 1) & ((1UL << 56) - 1)),
  _batchDepth   (batchDepth),
  _maxEntries   (maxEntries),
  _maxBatchSize (sizeof(Dgram) + maxEntries * maxSize),
  _batchBuffer  (new char[batchDepth * _maxBatchSize]),
  _datagrams    (new const Dgram*[batchDepth * maxEntries]),
  _pool         (Batch::size(), batchDepth),
  _batches      (new Batch*[batchDepth])
{
  if (__builtin_popcountl(duration) != 1)
  {
    fprintf(stderr, "Batch duration (%016lx) must be a power of 2\n",
            duration);
    abort();
  }

  Batch::init(_pool, _batchBuffer, _batchDepth, _maxBatchSize, _datagrams, maxEntries, _batches);

  // Revisit: Maybe make the following a dummy Batch that expires right away
  //  Then check if it's real before trying to post it
  //Dgram dg;
  //dg.seq = Sequence(ClockTime(0, 0), TimeStamp());

  _batch = NULL; // new(&_pool) Batch(_src, dg, dg.seq.stamp().pulseId());
}

BatchManager::~BatchManager()
{
  delete [] _batches;
  delete [] _datagrams;
  delete [] _batchBuffer;
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
  uint64_t pid = _startId(datagram->seq.stamp().pulseId());

  if (!_batch || _batch->expired(pid))
  {
    if (_batch)  post(_batch);

     // Resource wait if pool is empty
    _batch = new(&_pool) Batch(_src, *datagram, pid);
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
}

const Batch* BatchManager::batch(unsigned index, uint64_t id) const
{
  assert(index < _batchDepth);
  Batch* batch_ = _batches[index];
  assert(batch_->id() == id);
  return batch_;
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
