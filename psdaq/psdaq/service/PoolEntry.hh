/**
 * @file
 * @brief PoolEntry, the header placed in front of every buffer handed out by a Pool.
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

#ifndef PDS_POOLENTRY
#define PDS_POOLENTRY

#include <stddef.h>                     // for size_t

namespace Pds {

class Pool;           // Necessary to resolve forward reference...

/** Header in front of each pool buffer: two words used as queue links while the buffer is free, the owning Pool and a tag. The user buffer starts right after it. */
class PoolEntry
  {
  public:
    /** Record the owning pool and set the tag to 0xffffffff. */
    PoolEntry(Pool*);
    /** Placement new: return p (the size is not used). */
    void* operator new(size_t, char*);
    /** Return the PoolEntry header that precedes the user buffer buffer. */
    static PoolEntry* entry(void* buffer);
    void*         _opaque[2];  ///< Two words used as the Entry queue links while the buffer is on the free list of a GenericPool.
    Pool*         _pool;  ///< Pool that owns the buffer; Pool::free(void*) returns it there.
    unsigned long _tag;  ///< Set to 0xffffffff by the constructor; not read elsewhere in psdaq.
  protected:
    PoolEntry() {}
  };
}
/*
** ++
**
**
** --
*/

inline void* Pds::PoolEntry::operator new(size_t size, char* p)
  {
  return (void*)p;
  }

/*
** ++
**
**
** --
*/

inline Pds::PoolEntry::PoolEntry(Pds::Pool* pool):
  _pool(pool),
  _tag(0xffffffff)
  {
  }

/*
** ++
**
**
** --
*/

inline Pds::PoolEntry* Pds::PoolEntry::entry(void* buffer)
  {
  return (Pds::PoolEntry*)buffer - 1;
  }

#endif
