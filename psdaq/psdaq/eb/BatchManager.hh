#ifndef Pds_Eb_BatchManager_hh
#define Pds_Eb_BatchManager_hh

#include "eb.hh"
#include "Batch.hh"
#include "IndexPool.hh"
#include "psdaq/service/Fifo.hh"
#include "xtcdata/xtc/Dgram.hh"

#include <cstddef>                      // For size_t
#include <cstdint>                      // For uint64_t
#include <atomic>
#include <set>

namespace Pds {
  namespace Eb {

    using BatchFreeList = IndexPoolW<Batch>;
    using BatchBusyList = std::multiset<Batch*>;
    using AppPrm        = std::atomic<uintptr_t>;

    class BatchManager
    {
    public:
      BatchManager(size_t maxSize);
      ~BatchManager();
    public:
      void            stop();
      void            shutdown();
      void            flush();
      Batch*          fetch() const;
      Batch*          allocate(uint64_t pid);
      Batch*          allocate(const XtcData::Dgram& dg);
      void            release(const Batch*);
      Batch*          batch(unsigned idx);
      const Batch*    batch(unsigned idx) const;
      BatchBusyList&  busylist();
      size_t          maxSize()           const;
      size_t          maxBatchSize()      const;
      void*           batchRegion()       const;
      size_t          batchRegionSize()   const;
    public:
      void            dump()          const;
      int64_t         freeBatchCnt()  const;
      const uint64_t& batchAllocCnt() const;
      const uint64_t& batchFreeCnt()  const;
      const uint64_t& batchWaiting()  const;
    private:
      const size_t   _maxSize;       // Max size of the Dgrams to be batched
      const size_t   _maxBatchSize;  // Max batch size rounded up by page size
      char* const    _region;        // RDMA buffers for batches
      BatchFreeList  _freelist;      // Free list of Batch objects
      BatchBusyList  _busylist;      // List of in-use Batch objects
      AppPrm* const  _appPrms;       // Lookup array of application parameters
    private:
      Batch*         _batch;         // Batch currently being accumulated
    };
  };
};


inline
void Pds::Eb::BatchManager::stop()
{
  _freelist.stop();
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
Pds::Eb::Batch* Pds::Eb::BatchManager::batch(unsigned index)
{
  return &_freelist[index];
}

inline
const Pds::Eb::Batch* Pds::Eb::BatchManager::batch(unsigned index) const
{
  return &_freelist[index];
}

inline
size_t Pds::Eb::BatchManager::maxSize() const
{
  return _maxSize;
}

inline
size_t Pds::Eb::BatchManager::maxBatchSize() const
{
  return _maxBatchSize;
}

inline
void Pds::Eb::BatchManager::flush()
{
  _batch = nullptr;                     // Force a new batch to be started
}

inline
Pds::Eb::Batch* Pds::Eb::BatchManager::fetch() const
{
  return _batch;
}

inline
Pds::Eb::Batch* Pds::Eb::BatchManager::allocate(uint64_t pid)
{
  const auto idx = Pds::Eb::Batch::batchNum(pid);
  auto       bat = _freelist.allocate(idx);
  if (bat)  bat->initialize(pid);   // Full PID, not BatchNum

  _batch = bat;
  return bat;
}

inline
Pds::Eb::Batch* Pds::Eb::BatchManager::allocate(const XtcData::Dgram& dg)
{
  const auto pid = dg.seq.pulseId().value();
  const auto idx = Pds::Eb::Batch::batchNum(pid);

  if (_freelist.isAllocated(idx))
  {
    auto bat = batch(idx);

    assert((bat->id() & ~(BATCH_DURATION - 1)) == (pid & ~(BATCH_DURATION - 1)));

    bat->accumRogs(dg);

    return bat;
  }
  else
  {
    auto bat = _freelist.allocate(idx);

    bat->initialize(dg);

    _busylist.insert(bat);

    return bat;
  }
}

inline
Pds::Eb::BatchBusyList& Pds::Eb::BatchManager::busylist()
{
  return _busylist;
}

inline
void Pds::Eb::BatchManager::release(const Pds::Eb::Batch* batch)
{
  const_cast<Pds::Eb::Batch*>(batch)->release();
  _freelist.free(batch->index());
}

inline
int64_t Pds::Eb::BatchManager::freeBatchCnt() const
{
  return _freelist.numberofFreeObjects();
}

inline
const uint64_t& Pds::Eb::BatchManager::batchAllocCnt() const
{
  return _freelist.numberofAllocs();
}

inline
const uint64_t& Pds::Eb::BatchManager::batchFreeCnt() const
{
  return _freelist.numberofFrees();
}

inline
const uint64_t& Pds::Eb::BatchManager::batchWaiting() const
{
  return _freelist.waiting();
}

#endif
