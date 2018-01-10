#include "Reg.hh"

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include "DataDriver.h"

//#define DBUG

static int _fd = -1;

using namespace Kcu;

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

Reg& Reg::operator=(const unsigned r)
{
  uint32_t addr = reinterpret_cast<uintptr_t>(this);
  if (dmaWriteRegister(_fd, addr, r)<0)
    perror("Kcu::Reg write");
  return *this;
}

Reg::operator unsigned() const
{
  uint32_t addr = reinterpret_cast<uintptr_t>(this);
  uint32_t r;
  if (dmaReadRegister(_fd, addr, &r)<0)
    perror("Kcu::Reg read");
  return r;
}
