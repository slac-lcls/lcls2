/**
 * @file
 * @brief Batch, which tracks one batch of fixed-size entries in an RDMA region.
 */
#ifndef Pds_Eb_Batch_hh
#define Pds_Eb_Batch_hh

#include "eb.hh"

#include "psdaq/service/EbDgram.hh"

#include <cstdint>                      // For uint64_t
#include <cstddef>                      // for size_t


namespace Pds {
  namespace Eb {

    /** One batch of fixed-size entries in an RDMA region: its start pulse ID, its buffer (chosen from the pulse ID) and its current extent. */
    class Batch
    {
    public:
      /** Construct with zero entry size, ID and extent; the buffer pointer is left uninitialized. */
      Batch();
    public:
      /** Return pid modulo MAX_LATENCY (its low bits), used as the slot index of a batch. */
      static uint64_t index(uint64_t pid);
    public:
      /** Set the entry size to bufSize; returns 0. */
      int             initialize(size_t bufSize);
      /** Return the next entry (buffer plus current extent) as an EbDgram pointer and advance the extent by one entry size. There is no bounds check. */
      Pds::EbDgram*   allocate();      // Allocate buffer in batch
      /** Start the batch for pulse ID pid: the buffer becomes region plus index(pid) entries and the extent is reset to 0. Returns this. */
      Batch*          initialize(void* region, uint64_t pid);
      /** Return the current extent in bytes (number of allocated entries times the entry size). */
      size_t          extent() const;  // Current extent
      /** Return the full pulse ID the batch was initialized with. */
      uint64_t        id() const;      // Batch start pulse ID
      /** Return index(id()). */
      unsigned        index() const;   // Batch's index
      /** Return the start of this batch in the region. */
      const void*     buffer() const;  // Pointer to batch in RDMA space
      /** Print the batch and one line per entry (up to MAX_ENTRIES, stopping at an entry whose pulse ID is 0) to stderr, or a short message if the buffer is null. */
      void            dump() const;
    private:
      void*           _buffer;     // Pointer to RDMA space for this Batch
      size_t          _bufSize;    // Size of entries
      uint64_t        _id;         // Id of Batch, in case it remains empty
      unsigned        _extent;     // Current extent (unsigned is large enough)
    };
  };
};


inline
uint64_t Pds::Eb::Batch::index(uint64_t pid)
{
  return pid & (MAX_LATENCY - 1);
}

inline
unsigned Pds::Eb::Batch::index() const
{
  return index(_id);
}

inline
uint64_t Pds::Eb::Batch::id() const
{
  return _id;                           // Full PID, not BatchNum
}

inline
const void* Pds::Eb::Batch::buffer() const
{
  return _buffer;
}

inline
size_t Pds::Eb::Batch::extent() const
{
  return _extent;
}

inline
Pds::Eb::Batch* Pds::Eb::Batch::initialize(void* region, uint64_t pid)
{
  _id     = pid;                        // Full PID, not BatchNum
  _buffer = static_cast<char*>(region) + index(pid) * _bufSize;
  _extent = 0;
  //_buffer = region;                     // Revisit: For dense batch allocation idea

  return this;
}

inline
Pds::EbDgram* Pds::Eb::Batch::allocate()
{
  char* buf = static_cast<char*>(_buffer) + _extent;
  _extent += _bufSize;
  //if (_extent > (MAX_LATENCY - BATCH_DURATION) * _bufSize)  _extent = 0; // Revisit
  return reinterpret_cast<Pds::EbDgram*>(buf);
}

#endif
