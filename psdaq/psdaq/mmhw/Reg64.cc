#include "Reg64.hh"

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

using namespace Pds::Mmhw;

void Reg64::setBit  (unsigned b)
{
  uint64_t r = *this;
  *this = r | (1ULL<<b);
}

void Reg64::clearBit(unsigned b)
{
  uint64_t r = *this;
  *this = r &~(1ULL<<b);
}

Reg64& Reg64::operator=(const unsigned long r)
{
  _upper = r>>32;
  _lower = r&0xffffffff;
  return *this;
}

Reg64::operator unsigned long() const
{
    uint64_t r= unsigned(_upper);
    r <<= 32;
    r  |= unsigned(_lower);
    return r;
}
