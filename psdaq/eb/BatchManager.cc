#include "BatchManager.hh"
#include "Endpoint.hh"

#include "xtcdata/xtc/Dgram.hh"

using namespace XtcData;
using namespace Pds;
using namespace Pds::Eb;
using namespace Pds::Fabrics;

BatchManager::BatchManager(Src      src,
                           uint64_t duration, // = ~((1 << N) - 1) = 128 uS?
                           unsigned batchDepth,
                           unsigned maxEntries,
                           size_t   maxSize) :
  _src          (src),
  _duration     (duration),
  _durationShift(__builtin_ctzll(duration)),
  _durationMask (~((1 << __builtin_ctzll(duration)) - 1) & ((1UL << 56) - 1)),
  _maxBatchSize (maxEntries * maxSize),
  _pool         (Batch::size(), batchDepth)
{
  if (__builtin_popcountll(duration) != 1)
  {
    fprintf(stderr, "Batch duration (%016lx) must be a power of 2\n",
            duration);
    abort();
  }
}

BatchManager::~BatchManager()
{
}

void* BatchManager::batchPool() const
{
  return _pool.buffer();
}

size_t BatchManager::batchPoolSize() const
{
  return _pool.size();
}

void BatchManager::start(unsigned      batchDepth,
                         unsigned      maxEntries,
                         MemoryRegion* mr[2])
{
  Batch::init(_pool, batchDepth, maxEntries, mr);

  Dgram dg;
  dg.seq = Sequence(ClockTime(0, 0), TimeStamp());

  _batch = new(&_pool) Batch(_src, dg, dg.seq.stamp().pulseId());
}

void BatchManager::process(const Dgram* dg)
{
  uint64_t pid = _startId(dg->seq.stamp().pulseId());

  if (_batch->expired(pid))
  {
    post(_batch);

    _batch = new(&_pool) Batch(_src, *dg, pid); // Resource wait if pool is empty
  }

  _batch->append(*dg);
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
