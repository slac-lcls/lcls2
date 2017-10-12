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

class PoolEntry
  {
  public:
    PoolEntry(Pool*);
    void* operator new(size_t, char*);
    static PoolEntry* entry(void* buffer);
    void*         _opaque[2];
    Pool*         _pool;
    unsigned long _tag;
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
