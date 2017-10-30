#include "psdaq/hsd/LocalCpld.hh"

#include <unistd.h>
#include <stdio.h>

using namespace Pds::HSD;

#define I2C_READ(p) (p&0xff)

unsigned LocalCpld::revision() const { return I2C_READ(_reg[0]); }

unsigned LocalCpld::GAaddr  () const { return I2C_READ(_reg[1]); }

void     LocalCpld::reloadFpga() { _reg[2]=1; }

void     LocalCpld::GAaddr  (unsigned v) { _reg[1]=v; }

