#include "psdaq/hsd/OptFmc.hh"

#include <unistd.h>

using namespace Pds::HSD;

void OptFmc::resetPgp()
{
  unsigned v = this->qsfp;
  this->qsfp = v | (1<<4);  // assert qsfp reset
  usleep(10);
  this->qsfp = v;           // deassert qsfp reset
  usleep(1000);
  this->qsfp = v | (1<<5);  // assert tx reset
  usleep(10);
  this->qsfp = v;           // deassert tx reset
  usleep(1000);
  this->qsfp = v | (1<<6);  // assert rx reset
  usleep(10);
  this->qsfp = v;           // deassert rx reset
  usleep(1000);
}
