#include "GenericPoolW.hh"

#include <stdio.h>

using namespace Pds;

GenericPoolW::GenericPoolW(size_t sizeofObject, int numberofObjects) :
  GenericPool(sizeofObject, numberofObjects)//,
{
}

GenericPoolW::~GenericPoolW()
{
}

void* GenericPoolW::deque()
{
  Pds::PoolEntry* entry = removeW();
  return (void*)&entry[1];
}
