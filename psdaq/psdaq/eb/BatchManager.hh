#ifndef Pds_Eb_BatchManager_hh
#define Pds_Eb_BatchManager_hh

#include "Batch.hh"
#include "IndexPool.hh"

#include <cstddef>                      // For size_t
#include <cstdint>                      // For uint64_t
#include <atomic>

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
      BatchManager(uint64_t     duration,
                   unsigned     batchDepth,
                   unsigned     maxEntries,
                   size_t       maxSize);
      virtual ~BatchManager();
    public:
      virtual void post(const Batch*) = 0;
    public:
      void*        batchRegion() const;
      size_t       batchRegionSize() const;
      Batch*       allocate(const XtcData::Dgram*);
      void         deallocate(const Batch*);
      void         process(const XtcData::Dgram*, void* prm);
      void         flush();
      const Batch* batch(unsigned index) const;
      void         shutdown();
      uint64_t     batchId(uint64_t id) const;
      size_t       maxBatchSize() const;
    public:
      void         dump() const;
      int          freeBatchCount() const;
    private:
      uint64_t     _startId(uint64_t id) const;
    private:
      uint64_t     _duration;           // The lifetime of a batch (power of 2)
      uint64_t     _durationMask;       // Mask  off  insignificant bits
      unsigned     _batchDepth;         // Depth of the batch pool
      unsigned     _maxEntries;         // Max number of entries per batch
      size_t       _maxBatchSize;       // Max batch size rounded up to page boundary
      char*        _batchBuffer;        // RDMA buffers for batches
      BatchList    _batchFreelist;      // Free list of Batch objects
      AppPrm*      _appPrms;            // Lookup array of application free parameters
    private:
      Batch*       _batch;              // Batch currently being accumulated
    };
  };
};


inline
void* Pds::Eb::BatchManager::batchRegion() const
{
  return _batchBuffer;
}

inline
size_t Pds::Eb::BatchManager::batchRegionSize() const
{
  return _batchDepth * _maxBatchSize;
}

inline
const Pds::Eb::Batch* Pds::Eb::BatchManager::batch(unsigned index) const
{
  return &_batchFreelist[index];
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
uint64_t Pds::Eb::BatchManager::batchId(uint64_t id) const
{
  return id >> __builtin_ctzl(_duration); // Batch number
}

inline
uint64_t Pds::Eb::BatchManager::_startId(uint64_t id) const
{
  return id & _durationMask;
}

inline
void Pds::Eb::BatchManager::deallocate(const Pds::Eb::Batch* batch)
{
  _batchFreelist.free(const_cast<Batch*>(batch));
}

inline
int Pds::Eb::BatchManager::freeBatchCount() const
{
  return _batchFreelist.numberOfFreeObjects();
}

#endif
