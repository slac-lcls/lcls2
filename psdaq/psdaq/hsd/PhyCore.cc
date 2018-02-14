#include "psdaq/hsd/PhyCore.hh"

#include <stdio.h>

using namespace Pds::HSD;

void PhyCore::dump() const
{
  uint32_t* p = (uint32_t*)this;
  for(unsigned i=0x130/4; i<0x238/4; i++)
    printf("%08x%c", p[i], (i%8)==7?'\n':' ');
  printf("\n");
}
