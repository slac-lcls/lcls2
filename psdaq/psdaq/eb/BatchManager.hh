#ifndef Pds_Eb_BatchManager_hh
#define Pds_Eb_BatchManager_hh

#include "eb.hh"
#include "Batch.hh"
#include "xtcdata/xtc/Dgram.hh"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstddef>                      // For size_t
#include <cstdint>                      // For uint64_t
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace Pds {
  namespace Eb {

    using AppPrm_t = std::atomic<uintptr_t>;
    using lock_t   = std::mutex;
    using cv_t     = std::condition_variable_any;

    class BatchManager
    {
    public:
      BatchManager();
      ~BatchManager();
    public:
      int             initialize(size_t maxEntrySize, bool batching);
      void            stop();
      void            shutdown();
      Batch*          fetch(uint64_t pid);
      Batch*          fetchW(uint64_t pid);
      void            release(uint64_t pid);
      size_t          maxBatchSize()      const;
      void*           batchRegion()       const;
      size_t          batchRegionSize()   const;
      void            store(uint64_t pid, const void* appPrm);
      const void*     retrieve(uint64_t pid) const;
      bool            expired(uint64_t pid, uint64_t start) const;
    public:
      void            dump()          const;
      const uint64_t& batchWaiting()  const;
      const uint64_t& batchAllocCnt() const;
      const uint64_t& batchFreeCnt()  const;
      const uint64_t& inUseBatchCnt() const;
    private:
      size_t                _maxBatchSize; // Max batch size rounded up by page size
      size_t                _regSize;      // The allocated size of the _region
      char*                 _region;       // RDMA buffers for batches
      mutable lock_t        _lock;         // Resource wait lock
      mutable cv_t          _cv;           // Monitor the number of free batches
      std::vector<AppPrm_t> _appPrms;      // Lookup array of application parameters
      std::atomic<uint64_t> _lastFreed;    // PID of last freed batch
      Batch                 _batch;        // The currently being accumulated batch
      unsigned              _batching;     // Is batching enabled?
      std::atomic<uint64_t> _numAllocs;    // Number of Batch allocates
      std::atomic<uint64_t> _numFrees;     // Number of Batch frees
      std::atomic<uint64_t> _numInUse;     // Number of Batch allocates - frees
      mutable uint64_t      _nAllocs;      // Run-time monitoring version
      mutable uint64_t      _nFrees;       // Run-time monitoring version
      mutable uint64_t      _nInUse;       // Run-time monitoring version
      uint64_t              _waiting;      // State of allocation
      std::atomic<bool>     _terminate;    // Flag for breaking out of the cv wait
    };
  };
};


inline
void Pds::Eb::BatchManager::stop()
{
  {
    std::unique_lock<std::mutex> lock(_lock);
    _terminate.store(true, std::memory_order_release);
  }
  _cv.notify_one();
}

inline
void* Pds::Eb::BatchManager::batchRegion() const
{
  return _region;
}

inline
size_t Pds::Eb::BatchManager::batchRegionSize() const
{
  return MAX_BATCHES * _maxBatchSize;
}

inline
size_t Pds::Eb::BatchManager::maxBatchSize() const
{
  return _maxBatchSize;
}

inline
Pds::Eb::Batch* Pds::Eb::BatchManager::fetch(uint64_t pid)
{
  if (!_batching || expired(pid, _batch.id()))
  {
    _numAllocs.fetch_add(1, std::memory_order_acq_rel);
    _numInUse.fetch_add(1, std::memory_order_acq_rel);

    _batch.initialize(_region, pid);
  }
  return &_batch;
}

// @TODO: This wait could in principle go away if:
// - the TEB event timeout period is smaller than the latency buffer size
// - the TebContributor pending queue is changed to hold data Dgram batches
//   that are similarly constructed (time-wise) as the current Input batches
// In that case, overwriting the Input RDMA region after the TebContributor post
// is finished sending the buffer is not an issue.

inline
Pds::Eb::Batch* Pds::Eb::BatchManager::fetchW(uint64_t pid)
{
  if (!_batching || expired(pid, _batch.id()))
  {
    // Block when the head tries to pass the tail, unless empty or stopping
    {
      std::unique_lock<std::mutex> lock(_lock);
      ++_waiting;
      _cv.wait(lock, [&]{ return ((pid - _lastFreed.load(std::memory_order_acquire)) <
                                  (MAX_LATENCY - BATCH_DURATION))               ||
                                 (_numAllocs.load(std::memory_order_acquire) ==
                                  _numFrees.load(std::memory_order_acquire))    ||
                                 _terminate.load(std::memory_order_acquire); });
      --_waiting;
      if (_terminate.load(std::memory_order_acquire))  return nullptr;
      _numAllocs.fetch_add(1, std::memory_order_acq_rel);
      _numInUse.fetch_add(1, std::memory_order_acq_rel);
    }

    _batch.initialize(_region, pid);
  }
  return &_batch;
}

inline
void Pds::Eb::BatchManager::release(uint64_t pid)
{
  std::unique_lock<std::mutex> lock(_lock);
  _lastFreed.store(pid, std::memory_order_release);
  _cv.notify_one();
  _numFrees.fetch_add(1, std::memory_order_acq_rel);
  _numInUse.fetch_sub(1, std::memory_order_acq_rel);
}

inline
void Pds::Eb::BatchManager::store(uint64_t pid, const void* appPrm)
{
  unsigned idx = pid & (MAX_LATENCY - 1);

  _appPrms[idx] = reinterpret_cast<uintptr_t>(appPrm);
}

inline
const void* Pds::Eb::BatchManager::retrieve(uint64_t pid) const
{
  unsigned idx = pid & (MAX_LATENCY - 1);

  return reinterpret_cast<void*>((uintptr_t)_appPrms[idx]);
}

inline
bool Pds::Eb::BatchManager::expired(uint64_t pid, uint64_t start) const
{
  return (pid ^ start) & ~(BATCH_DURATION - 1);
}

inline
const uint64_t& Pds::Eb::BatchManager::inUseBatchCnt() const
{
  //_nInUse = batchAllocCnt() - batchFreeCnt();
  _nInUse = _numInUse.load(std::memory_order_relaxed);
  return _nInUse;
}

inline
const uint64_t& Pds::Eb::BatchManager::batchAllocCnt() const
{
  _nAllocs = _numAllocs.load(std::memory_order_relaxed);
  return _nAllocs;
}

inline
const uint64_t& Pds::Eb::BatchManager::batchFreeCnt() const
{
  _nFrees = _numFrees.load(std::memory_order_relaxed);
  return _nFrees;
}

inline
const uint64_t& Pds::Eb::BatchManager::batchWaiting() const
{
  return _waiting;
}

#endif
