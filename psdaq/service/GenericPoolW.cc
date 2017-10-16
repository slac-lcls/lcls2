#include "GenericPoolW.hh"

#include <stdio.h>

using namespace Pds;

GenericPoolW::GenericPoolW(size_t sizeofObject, int numberofObjects) :
  GenericPool(sizeofObject, numberofObjects),
  _sem(Semaphore::EMPTY)
{
  //
  //  Note that the base class GenericPool populates the pool by
  //  calling enque() before we have overridden that function.
  //
  for(int i=0; i<numberofObjects; i++)
    _sem.give();
}

GenericPoolW::~GenericPoolW()
{
}

void* GenericPoolW::deque()
{
  _sem.take();
  void* p = GenericPool::deque();
  while( p == NULL ) {  // this should never happen
    printf("GenericPoolW::deque returned NULL with depth %d.  Depleting semaphore.\n", depth());
    _sem.take();
    p = GenericPool::deque();
  }
  return p;
}

void GenericPoolW::enque(PoolEntry* entry)
{
  GenericPool::enque(entry);
  _sem.give();
}
