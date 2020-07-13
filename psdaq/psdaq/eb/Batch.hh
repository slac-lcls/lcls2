#ifndef Pds_Eb_Batch_hh
#define Pds_Eb_Batch_hh

#include "eb.hh"

#include "psdaq/service/EbDgram.hh"

#include <cstdint>                      // For uint64_t
#include <cstddef>                      // for size_t


namespace Pds {
  namespace Eb {

    class Batch
    {
    public:
      Batch();
    public:
      static uint64_t index(uint64_t pid);
    public:
      int             initialize(size_t bufSize);
      Pds::EbDgram*   allocate();      // Allocate buffer in batch
      Batch*          initialize(void* region, uint64_t pid);
      size_t          extent() const;  // Current extent
      uint64_t        id() const;      // Batch start pulse ID
      unsigned        index() const;   // Batch's index
      const void*     buffer() const;  // Pointer to batch in RDMA space
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
