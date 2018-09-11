#ifndef Pds_Eb_BatchManager_hh
#define Pds_Eb_BatchManager_hh

#include "Batch.hh"

#include "psdaq/service/Fifo.hh"

#include <stdlib.h>
#include <stdint.h>
#include <cstddef>
#include <string>
#include <vector>
#include <atomic>

namespace XtcData {
  class Dgram;
};

namespace Pds {
  namespace Eb {

    using BatchFifoW  = FifoW<Batch*>;
    using BatchVector = std::vector<Batch>;

    class BatchManager
    {
    public:
      BatchManager(uint64_t     duration,
                   unsigned     batchDepth,
                   unsigned     maxEntries,
                   size_t       maxSize);
      virtual ~BatchManager();
    public:
      virtual void post(const Batch*)          = 0;
      virtual void post(const XtcData::Dgram*) = 0;
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
      uint64_t     _durationShift;      // Shift away insignificant bits
      uint64_t     _durationMask;       // Mask  off  insignificant bits
      unsigned     _batchDepth;         // Depth of the batch pool
      unsigned     _maxEntries;         // Max number of entries per batch
      size_t       _maxBatchSize;       // Max batch size rounded up to page boundary
      char*        _batchBuffer;        // RDMA buffers for batches
      BatchFifoW   _batchFreeList;      // Free list of Batch objects
      BatchVector  _batches;            // Lookup array of batches
      std::atomic<uintptr_t>* _appPrms; // Lookup array of application free parameters
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
  return (index < _batchDepth) ? &_batches[index] : nullptr;
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

//inline
//uint64_t Pds::Eb::BatchManager::batchId(uint64_t id) const
//{
//  return id / _duration;         // Batch number
//}

inline
uint64_t Pds::Eb::BatchManager::batchId(uint64_t id) const
{
  return id >> _durationShift;          // Batch number
}

//inline
//uint64_t Pds::Eb::BatchManager::_startId(uint64_t id) const
//{
//  return batchId(id) * _duration;     // Current batch ID
//}

inline
uint64_t Pds::Eb::BatchManager::_startId(uint64_t id) const
{
  return id & _durationMask;
}

inline
void Pds::Eb::BatchManager::deallocate(const Pds::Eb::Batch* batch)
{
  _batchFreeList.push(const_cast<Batch*>(batch));
}

inline
int Pds::Eb::BatchManager::freeBatchCount() const
{
  return _batchFreeList.count();
}

#endif
