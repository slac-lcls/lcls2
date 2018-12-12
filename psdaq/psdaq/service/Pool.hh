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

#define PoolDeclare                                                     \
  void* operator new   (size_t size,                                    \
                        Pool*  pool)   { return pool->alloc(size); }    \
  void  operator delete(void*  buffer) { Pool::free(buffer); }


namespace Pds {
class Pool
  {
  public:
    virtual ~Pool();
    Pool(size_t sizeofObject, int numberofOfObjects);
    Pool(size_t sizeofObject, int numberofOfObjects, unsigned alignBoundary);
    void*           alloc(size_t size);
    virtual void    free(PoolEntry*);
    size_t          sizeofObject()              const;
    int             numberofObjects()           const;
    const uint64_t& numberofAllocs()            const;
    const uint64_t& numberofFrees()             const;
    int             numberOfAllocatedObjects()  const;
    int             numberOfFreeObjects()       const;
  public:
    static void     free(void* buffer);
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
