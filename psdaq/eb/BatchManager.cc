#include "BatchManager.hh"
#include "EbFtBase.hh"

#include "xtcdata/xtc/Dgram.hh"

using namespace XtcData;
using namespace Pds;
using namespace Pds::Eb;
using namespace Pds::Fabrics;

BatchManager::BatchManager(EbFtBase& outlet,
                           unsigned  id,       // Revisit: Should be a Src?
                           uint64_t  duration, // = ~((1 << N) - 1) = 128 uS?
                           unsigned  batchDepth,
                           unsigned  maxEntries,
                           size_t    contribSize) :
  _src          (id),
  _duration     (duration),
  _durationShift(__builtin_ctzll(duration)),
  _durationMask (~((1 << __builtin_ctzll(duration)) - 1)),
  _maxBatchSize (maxEntries * contribSize),
  _pool         (Batch::size(), batchDepth),
  _inFlightList (),
  _inFlightLock (),
  _outlet       (outlet)
{
  if (__builtin_popcountll(duration) != 1)
  {
    fprintf(stderr, "Batch duration (%016lx) must be a power of 2\n",
            duration);
    abort();
  }

  printf("Dumping pool 1:\n");
  _pool.dump();

  Batch::init(_pool, batchDepth, maxEntries);

  printf("Dumping pool 2:\n");
  _pool.dump();

  Dgram     dg; //(TypeId(), Src());
  dg.seq = Sequence(ClockTime(0, 0), TimeStamp());

  _batch = new(&_pool) Batch(dg, dg.seq.stamp().pulseId());
}

BatchManager::~BatchManager()
{
}

void BatchManager::process(const Dgram* contrib, void* arg)
{
  uint64_t id = _startId(contrib->seq.stamp().pulseId());

  if (_batch->expired(id))
  {
    post(_batch, arg);

    _batch = new(&_pool) Batch(*contrib, id);
  }

  _batch->append(*contrib);
}

void BatchManager::postTo(Batch*   batch,
                          unsigned dst,
                          unsigned slot)
{
  uint64_t dstOffset = slot * _maxBatchSize;

  _outlet.post(batch->finalize(), dst, dstOffset, NULL);

  // Revisit: No obvious need to wait for completion here as nothing can be done
  // with this batch or its remote instance until a result is sent
  // - This is true on the contributor side
  // - Revisit for the EB result side

  _inFlightList.insert(batch);          // Revisit: Replace with atomic list
}

void BatchManager::release(uint64_t id)
{
  Batch* batch = _inFlightList.atHead();
  Batch* end   = _inFlightList.empty();
  while (batch != end)
  {
    Entry* next = batch->next();

    if (id > batch->id())
    {
      _inFlightList.remove(batch);      // Revisit: Replace with atomic list
      delete batch;
    }
    batch = (Batch*)next;
  }
}

void BatchManager::shutdown()
{
  _outlet.shutdown();
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
