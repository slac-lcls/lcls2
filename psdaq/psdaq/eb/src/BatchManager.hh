/**
 * @file
 * @brief BatchManager, which owns the region where a TEB contributor builds its batches.
 */
#ifndef Pds_Eb_BatchManager_hh
#define Pds_Eb_BatchManager_hh

#include "eb.hh"

#include <cstddef>                      // For size_t
#include <cstdint>                      // For uint64_t

namespace Pds {
  namespace Eb {

    /** Owns the page-aligned region in which batches are built and decides when a batch has expired. */
    class BatchManager
    {
    public:
      /** Construct with no region. */
      BatchManager();
      /** Free the region. */
      ~BatchManager();
    public:
      /** Reallocate the region for numBatches batches of maxEntries entries of maxEntrySize bytes if that total size changed, and set the expiration mask from maxEntries. Returns 0, 1 if maxEntries is not a power of 2, or ENOMEM if allocation fails. */
      int    initialize(size_t maxEntrySize, unsigned maxEntries, unsigned numBatches);
      /** Zero the whole region; it is not freed. */
      void   shutdown();
      /** Return the address of entry idx: the region start plus idx times the entry size. */
      void*  fetch(unsigned idx) const;
      /** Return the start of the region. */
      void*  batchRegion() const;
      /** Return the size of the region in bytes. */
      size_t batchRegionSize() const;
      /** Return true if pid and start differ in any bit above the low log2(maxEntries) bits, that is, if they lie in different batch periods. */
      bool   expired(uint64_t pid, uint64_t start) const;
    public:
      /** Print the region base, region size, maximum batch size and entry size to stderr. */
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
void* Pds::Eb::BatchManager::fetch(unsigned index) const
{
  return _region  + index * _maxEntrySize;
}


inline
bool Pds::Eb::BatchManager::expired(uint64_t pid, uint64_t start) const
{
  return (pid ^ start) & _mask;
}

#endif
