#include "psdaq/hsd/I2cSwitch.hh"

#include <stdio.h>

using namespace Pds::HSD;

void I2cSwitch::dump() const
{
  printf("I2C Switch: %x\n", (unsigned)_control&0xff);
}

void I2cSwitch::select(Port p)
{
  _control = unsigned(p);
}
