/**
 * @file
 * @brief Pool, the abstract base of fixed-size buffer pools, and the PoolDeclare macro that routes a class's new and delete through a pool.
 */
/*
** ++
**  Package:
**  Service
**
**  Abstract:
**
**  Author:
**      Michael Huffer, SLAC, (415) 926-4269
**
**  Creation Date:
**  000 - December 20 1,1997
**
**  Revision History:
**  None.
**
** --
*/

#ifndef PDS_POOL
#define PDS_POOL


#include "PoolEntry.hh"

#include <cstdint>

/** Declares an operator new(size, Pool*) that allocates from the pool and an operator delete that returns the buffer with Pool::free(); put it in a class body to allocate that class from a Pool. */
#define PoolDeclare                                                     \
  void* operator new   (size_t size,                                    \
                        Pool*  pool)   { return pool->alloc(size); }    \
  void  operator delete(void*  buffer) { Pool::free(buffer); }


namespace Pds {
/** Abstract pool of numberofObjects equal buffers, each preceded by a PoolEntry, with allocation and free counters. Subclasses provide the storage and the free list. */
class Pool
  {
  public:
    /** Does nothing; empty virtual destructor. */
    virtual ~Pool();
    /** Set the object size and count and round each allocation (object plus PoolEntry header) up to a multiple of the PoolEntry size; the subclass calls populate() to create the buffers. */
    Pool(size_t sizeofObject, int numberofOfObjects);
    /** Like Pool(sizeofObject, numberofOfObjects), but rounds each allocation up to a multiple of alignBoundary. */
    Pool(size_t sizeofObject, int numberofOfObjects, unsigned alignBoundary);
    /** Take a buffer from the free list and count the allocation; returns nullptr if size exceeds the object size or none is available. */
    void*           alloc(size_t size);
    /** Return entry to the free list (enque()). */
    virtual void    free(PoolEntry*);
    /** Zero the allocation and free counters. */
    void            clearCounters();
    /** Return the object size given to the constructor. */
    size_t          sizeofObject()              const;
    /** Return the number of buffers. */
    int             numberofObjects()           const;
    /** Return a reference to the allocation counter. */
    const uint64_t& numberofAllocs()            const;
    /** Return a reference to the free counter. */
    const uint64_t& numberofFrees()             const;
    /** Return allocations minus frees. */
    int             numberOfAllocatedObjects()  const;
    /** Return the number of buffers minus numberOfAllocatedObjects(). */
    int             numberOfFreeObjects()       const;
  public:
    /** Return the user buffer buffer to the pool recorded in its PoolEntry header and count the free. */
    static void     free(void* buffer);
    /** Return numberOfFreeObjects() of the pool that owns buffer. */
    static int      numberOfFreeObjects(void* buffer);
  protected:
    size_t          sizeofAllocate()  const;
    virtual void*   deque()               = 0;
    virtual void    enque(PoolEntry*)     = 0;
    virtual void*   allocate(size_t size) = 0;
    void            populate();
  private:
    int             _numberofObjects;
    int             _remaining;
    size_t          _sizeofObject;
    size_t          _quanta;
    uint64_t        _numberofAllocs;
    uint64_t        _numberofFrees;
  };
}
/*
** ++
**
**
** --
*/

inline void* Pds::Pool::alloc(size_t size)
  {
  void* p = (size > _sizeofObject) ? (void*)0 : deque();
  if (p)
    _numberofAllocs++;
  return p;
  }

/*
** ++
**
**
** --
*/

inline void Pds::Pool::free(void* buffer)
  {
  Pds::Pool* pool = (Pds::PoolEntry::entry(buffer))->_pool;
  pool->free(Pds::PoolEntry::entry(buffer));
  pool->_numberofFrees++;
  }

inline int Pds::Pool::numberOfFreeObjects(void* buffer)
  {
  Pds::Pool* pool = (Pds::PoolEntry::entry(buffer))->_pool;
  return pool->numberOfFreeObjects();
  }

/*
** ++
**
**
** --
*/

inline Pds::Pool::~Pool()
  {
  }

/*
** ++
**
**
** --
*/

inline size_t Pds::Pool::sizeofObject() const
  {
  return _sizeofObject;
  }

/*
** ++
**
**
** --
*/

inline size_t Pds::Pool::sizeofAllocate() const
  {
  return _quanta;
  }

/*
** ++
**
**
** --
*/

inline int Pds::Pool::numberofObjects() const
  {
  return _numberofObjects;
  }

/*
** ++
**
**
** --
*/

inline const uint64_t& Pds::Pool::numberofAllocs() const
  {
  return _numberofAllocs;
  }

/*
** ++
**
**
** --
*/

inline const uint64_t& Pds::Pool::numberofFrees() const
  {
  return _numberofFrees;
  }

inline int Pds::Pool::numberOfAllocatedObjects() const
{
   return _numberofAllocs - _numberofFrees;
}

inline int Pds::Pool::numberOfFreeObjects() const
{
   return _numberofObjects - numberOfAllocatedObjects();
}

#endif
