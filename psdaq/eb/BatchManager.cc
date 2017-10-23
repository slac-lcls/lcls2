#include "BatchManager.hh"
#include "FtOutlet.hh"

#include "psdaq/xtc/Datagram.hh"

using namespace XtcData;
using namespace Pds;
using namespace Pds::Eb;
using namespace Pds::Fabrics;

static uint64_t clkU64(const ClockTime& clk)
{
  return uint64_t(clk.seconds()) << 32 | uint64_t(clk.nanoseconds());
}

BatchManager::BatchManager(FtOutlet& outlet,
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
  ClockTime clk(0, 0);
  dg.seq = Sequence(clk, TimeStamp());

  _batch = new(&_pool) Batch(dg, clk);
}

BatchManager::~BatchManager()
{
}

void BatchManager::process(const Datagram* contrib, void* arg)
{
  ClockTime start = _startTime(contrib->seq.clock());

  if (_batch->expired(start))
  {
    post(_batch, arg);

    _batch = new(&_pool) Batch(*contrib, start);
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

void BatchManager::release(const ClockTime& time)
{
  Batch* batch = _inFlightList.atHead();
  Batch* end   = _inFlightList.empty();
  while (batch != end)
  {
    Entry* next = batch->next();

    if (time > batch->clock())
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

//uint64_t BatchManager::batchId(const ClockTime & clk) const
//{
//  return clk.u64() / _duration;         // Batch number since EPOCH
//}

uint64_t BatchManager::batchId(const ClockTime& clk) const
{
  return clkU64(clk) >> _durationShift;   // Batch number since EPOCH
}

//uint64_t BatchManager::_startTime(const ClockTime& clk) const
//{
//  return _batchId(clk) * _duration;     // Current batch start time
//}

const ClockTime BatchManager::_startTime(const ClockTime& clk) const
{
  uint64_t u64(clkU64(clk) & _durationMask);
  return ClockTime((u64 >> 32) & 0xffffffffull, u64 & 0xffffffffull);
}
