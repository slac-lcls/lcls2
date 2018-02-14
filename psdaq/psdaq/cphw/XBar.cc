#include "psdaq/cphw/XBar.hh"

#include <stdio.h>

using namespace Pds::Cphw;

void XBar::setOut( Map out, Map in )
{
  outMap[unsigned(out)] = unsigned(in);
}

void XBar::dump  () const
{
  static const char* src[] = {"EVR0", "FPGA", "BP", "EVR1"};
  for(unsigned i=0; i<4; i++)
    printf("OUT[%u (%4s)] = %u (%4s)\n", i, src[i], unsigned(outMap[i]), src[outMap[i]]);
}
