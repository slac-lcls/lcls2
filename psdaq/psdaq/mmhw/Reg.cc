#include "Reg.hh"

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include "DataDriver.h"

//#define DBUG

static int _fd = -1;
static bool _verbose = false;

using namespace Pds::Mmhw;

void Reg::setBit  (unsigned b)
{
  unsigned r = *this;
  *this = r | (1<<b);
}

void Reg::clearBit(unsigned b)
{
  unsigned r = *this;
  *this = r &~(1<<b);
}

void Reg::set(unsigned fd)
{
  _fd = fd;
}

void Reg::verbose(bool v)
{
  _verbose = v;
}

Reg& Reg::operator=(const unsigned r)
{
  uint32_t addr = reinterpret_cast<uintptr_t>(this);
  if (_verbose)
      printf("Write [0x%x] : 0x%x\n",addr,r);

  if (dmaWriteRegister(_fd, addr, r)<0)
    perror("Pds::Mmhw::Reg write");

  return *this;
}

Reg::operator unsigned() const
{
  uint32_t addr = reinterpret_cast<uintptr_t>(this);
  uint32_t r=-1UL;
  if (dmaReadRegister(_fd, addr, &r)<0) {
      printf("read 0x%x\n",addr);
      perror("Pds::Mmhw::Reg read");
  }

  if (_verbose)
      printf("Read [0x%x] : 0x%x\n", addr, r);
  return r;
}

