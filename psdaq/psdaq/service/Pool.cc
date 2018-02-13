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

#include "Pool.hh"

using namespace Pds;

/*
** ++
**
**
** --
*/

Pool::Pool(size_t sizeofObject, int numberofObjects) :
  _sizeofObject(sizeofObject),
  _numberofObjects(numberofObjects),
  _numberofAllocs(0),
  _numberofFrees(0)
  {
  size_t quanta      = _sizeofObject + sizeof(PoolEntry);
  unsigned remainder = quanta % sizeof(PoolEntry);

  if(remainder) quanta += sizeof(PoolEntry) - remainder;

  _quanta    = quanta;
  _remaining = _numberofObjects;
  }

Pool::Pool(size_t sizeofObject, int numberofObjects, unsigned alignBoundary) :
  _sizeofObject(sizeofObject),
  _numberofObjects(numberofObjects),
  _numberofAllocs(0),
  _numberofFrees(0)
  {
  size_t quanta      = _sizeofObject + sizeof(PoolEntry);
  unsigned remainder = quanta % alignBoundary;

  if(remainder) quanta += alignBoundary - remainder;

  _quanta    = quanta;
  _remaining = _numberofObjects;
  }

/*
** ++
**
**
** --
*/

void Pool::populate()
  {
  int remaining = _remaining;
  if(!remaining) return;

  PoolEntry* entry;
  char* buffer;

  do {
    if ((buffer = (char*) allocate(_quanta)))
      {
        entry = new(buffer) PoolEntry(this);
        enque(entry);
      }
  } while (--remaining);

  }


void Pds::Pool::free(PoolEntry* entry)
{
  enque(entry);
}

