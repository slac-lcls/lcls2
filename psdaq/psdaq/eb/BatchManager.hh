#ifndef Pds_Eb_BatchManager_hh
#define Pds_Eb_BatchManager_hh

#include "eb.hh"
#include "Batch.hh"
#include "IndexPool.hh"
#include "psdaq/service/Fifo.hh"

#include <cstddef>                      // For size_t
#include <cstdint>                      // For uint64_t
#include <atomic>
#include <chrono>

namespace XtcData {
  class Dgram;
};

namespace Pds {
  namespace Eb {

    using BatchList = IndexPoolW<Batch>;
    using AppPrm    = std::atomic<uintptr_t>;

    class BatchManager
    {
    public:
      BatchManager(size_t maxSize);
      ~BatchManager();
    public:
      void            shutdown();
      void            flush();
      Batch*          fetch() const;
      Batch*          allocate(uint64_t pid);
      void            release(const Batch*);
      Batch*          batch(unsigned idx);
      const Batch*    batch(unsigned idx) const;
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
      BatchList      _batchFreelist; // Free list of Batch objects
      AppPrm* const  _appPrms;       // Lookup array of application parameters
    private:
      Batch*         _batch;         // Batch currently being accumulated
    };
  };
};


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
  return &_batchFreelist[index];
}

inline
const Pds::Eb::Batch* Pds::Eb::BatchManager::batch(unsigned index) const
{
  return &_batchFreelist[index];
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
  const auto tmo(std::chrono::milliseconds(5000));
  const auto id (Pds::Eb::Batch::batchId(pid));
  Pds::Eb::Batch* batch = _batchFreelist.allocate(id, tmo);
  if (batch)  batch->initialize(pid);
  _batch = batch;
  return batch;
}

inline
void Pds::Eb::BatchManager::release(const Pds::Eb::Batch* batch)
{
  const_cast<Pds::Eb::Batch*>(batch)->release();
  _batchFreelist.free(batch->index());
}

inline
int64_t Pds::Eb::BatchManager::freeBatchCnt() const
{
  return _batchFreelist.numberofFreeObjects();
}

inline
const uint64_t& Pds::Eb::BatchManager::batchAllocCnt() const
{
  return _batchFreelist.numberofAllocs();
}

inline
const uint64_t& Pds::Eb::BatchManager::batchFreeCnt() const
{
  return _batchFreelist.numberofFrees();
}

inline
const uint64_t& Pds::Eb::BatchManager::batchWaiting() const
{
  return _batchFreelist.waiting();
}

#endif
