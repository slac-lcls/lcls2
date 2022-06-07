#ifndef Pds_Eb_BatchManager_hh
#define Pds_Eb_BatchManager_hh

#include "eb.hh"

#include <cstddef>                      // For size_t
#include <cstdint>                      // For uint64_t

namespace Pds {
  namespace Eb {

    class BatchManager
    {
    public:
      BatchManager();
      ~BatchManager();
    public:
      int    initialize(size_t maxEntrySize, unsigned maxEntries, unsigned numBatches);
      void   shutdown();
      void*  fetch(unsigned idx);
      void*  batchRegion()     const;
      size_t batchRegionSize() const;
      bool   expired(uint64_t pid, uint64_t start) const;
    public:
      void   dump() const;
    private:
      size_t   _regSize;      // The allocated size of the _region
      char*    _region;       // RDMA buffers for batches
      uint64_t _mask;         // PulseId expiration mask
      size_t   _maxEntrySize; // Space reserved for a batch entry
      size_t   _maxBatchSize; // Max batch size rounded up by page size
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
  return _regSize;
}

inline
void* Pds::Eb::BatchManager::fetch(unsigned index)
{
  return _region  + index * _maxEntrySize;
}


inline
bool Pds::Eb::BatchManager::expired(uint64_t pid, uint64_t start) const
{
  return (pid ^ start) & _mask;
}

#endif
