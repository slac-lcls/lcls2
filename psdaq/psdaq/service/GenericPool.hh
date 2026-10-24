/**
 * @file
 * @brief GenericPool, a Pool whose buffers come from one contiguous allocation and whose free list is a locked Queue.
 */
/*
** ++
**  Package:
**	Service
**
**  Abstract:
**      
**  Author:
**      Michael Huffer, SLAC, (415) 926-4269
**
**  Creation Date:
**	000 - December 20 1,1997
**
**  Revision History:
**	None.
**
** --
*/

#ifndef PDS_GENERICPOOL
#define PDS_GENERICPOOL

#include "Pool.hh"
#include "Queue.hh"

namespace Pds {
/** Pool whose buffers are carved from one new[] block and kept on a Queue of PoolEntry (spin-locked); allocation returns nullptr when the queue is empty. */
class GenericPool : public Queue<PoolEntry>, public Pool
  {
  public:
    /** Allocate a block for numberofObjects buffers of sizeofObject bytes plus headers and put all buffers on the free list. */
    GenericPool(size_t sizeofObject, int numberofObjects);
    /** Like GenericPool(sizeofObject, numberofObjects), with each user buffer aligned to alignBoundary (the block gets alignBoundary extra bytes). */
    GenericPool(size_t sizeofObject, int numberofObjects, unsigned alignBoundary);
    /** Free the block. */
    ~GenericPool();
  public:
    /** Return the start of the block. */
    void*  buffer() const;
    /** Return the block size in bytes. */
    size_t size  () const;
  protected:
    virtual void* deque(); 
    virtual void  enque(PoolEntry*);
    virtual void* allocate(size_t size);
  public:
    /** Print the block size, address and fill offset, the object size and count, and the counters to stdout. */
    void dump() const;
  private:
    size_t _bounds;
    char* _buffer;
    size_t _current;
  };
}
/*
** ++
**
**
** --
*/

inline
Pds::GenericPool::GenericPool(size_t sizeofObject, int numberofObjects) :
  Pds::Pool(sizeofObject, numberofObjects),
  _bounds(sizeofAllocate()*numberofObjects),
  _buffer(new char[_bounds]),
  _current(0)
{
populate();
}

/*
** ++
**   A constructor which provides aligned memory accesses
**
** --
*/

inline
Pds::GenericPool::GenericPool(size_t sizeofObject, int numberofObjects, unsigned alignBoundary) :
  Pds::Pool(sizeofObject, numberofObjects, alignBoundary),
  _bounds(sizeofAllocate()*numberofObjects+alignBoundary),
  _buffer(new char[_bounds]),
  _current(alignBoundary-(((size_t)_buffer+sizeof(PoolEntry))%alignBoundary))
{
populate();
}

/*
** ++
**
**
** --
*/

inline Pds::GenericPool::~GenericPool()
  {
  delete[] _buffer;
  }

inline void*  Pds::GenericPool::buffer() const
{
  return _buffer;
}

inline size_t Pds::GenericPool::size() const
{
  return _bounds;
}

/*
** ++
**
**
** --
*/

inline void* Pds::GenericPool::deque() 
  {
  Pds::PoolEntry* entry = remove();
  return (entry != empty()) ? (void*)&entry[1] : (void*)0;
  }

/*
** ++
**
**
** --
*/

inline void Pds::GenericPool::enque(PoolEntry* entry) 
  {
  insert(entry);
  }

/*
** ++
**
**
** --
*/

inline void* Pds::GenericPool::allocate(size_t size)
  {
  size_t offset  = _current;
  char* entry = _buffer + offset;

  if ((offset += size) <= _bounds)
    {
    _current = offset;
    return (void*) entry;
    }
  else
    {
    return (void*) 0;
    }  
  }

#endif
