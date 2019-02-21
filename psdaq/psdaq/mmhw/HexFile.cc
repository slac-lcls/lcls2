#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include "psdaq/mmhw/HexFile.hh"
#include "psdaq/mmhw/McsFile.hh"

using namespace Pds::Mmhw;

HexFile::HexFile(const char* name)
{
  FILE* f = fopen(name,"r");
  if (!f) {
    perror("open");
    exit(1);
  }

  uint8_t d;
  unsigned i=0;
  int c;
  while( (c=fgetc(f))!= EOF ) {
    d <<= 4;
    if (isdigit(c)) 
      d |= c-'0';
    else if (isxdigit(c))
      d |= c-'a'+10;
    else {
      printf("Unexpected character %u [%c] at %u\n",
             c, (char)c, i);
      break;
    }
    i++;
    if ((i&1)==0)
      _dataList.push_back(d);
  }

  fclose(f);
}

HexFile::HexFile(const McsFile& mcs, unsigned addr)
{
  unsigned maddr=0;
  unsigned mdata=0;
  for(unsigned i=0; maddr < mcs.endAddr(); i++) {
    mcs.entry(i,maddr,mdata);
    if (maddr < addr) continue;
    while(addr < maddr) {
      _dataList.push_back(0xff);
      addr++;
    }
    _dataList.push_back(mdata);
    addr++;
  }
}
