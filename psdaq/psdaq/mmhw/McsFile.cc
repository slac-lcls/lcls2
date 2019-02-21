#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include "psdaq/mmhw/McsFile.hh"

using namespace Pds::Mmhw;

McsFile::McsFile(const char* name)
{
  FILE* f = fopen(name,"r");
  if (!f) {
    perror("open");
    exit(1);
  }

  unsigned baseAddr=0, address=0;

  const unsigned maxchar=256;
  char linePtr[maxchar];
  char* lptr = &linePtr[0];
  size_t nchar=maxchar;
  ssize_t n;
  unsigned iline=0;
  while( (n=getline(&lptr, &nchar, f)) ) {
    if (--n < 5) continue;

    linePtr[n]=0;
    if (linePtr[0]!=':') {
      printf("Missing start code. Line[%d]: [%s]\n",
             iline, linePtr);
      exit(1);
    }
    unsigned byteCnt, addr, recordType;
    sscanf(&linePtr[1],"%02x%04x%02x",&byteCnt,&addr,&recordType);

    if (byteCnt>16) {
      printf("Invalid byte count %u. [%s]\n", byteCnt, linePtr);
      exit(1);
    }

    unsigned cs = 0;
    unsigned v;
    for(unsigned i=0; i<byteCnt+4; i++) {
      sscanf(&linePtr[2*i+1],"%02x",&v);
      cs += v;
    }
    cs &= 0xff;
    sscanf(&linePtr[n-2],"%02x",&v);
    v = ~(v-1)&0xff;

    if (cs!=v) {
      printf("Bad checksum: line[%u]  Sum: %x  Chksum: %x  [%s]\n",
             iline, cs, v, linePtr);
      exit(1);
    }

    if (recordType==0) {
      if (byteCnt==0) {
        printf("Invalid byte count %u  [%s]\n", byteCnt, linePtr);
        exit(1);
      }
      for(unsigned i=0; i<byteCnt; i++) {
        sscanf(&linePtr[9+2*i],"%02x",&v);
        address = baseAddr + addr + i;
        _addrList.push_back(address);
        _dataList.push_back(v);
      }
      _endAddr = address;
    }
    else if (recordType==1) {
      break;
    }
    else if (recordType==4) {
      if (byteCnt != 2) {
        printf("Byte count %u must be 2 for ELA records [%s]\n",
               byteCnt, linePtr);
        exit(1);
      }
      if (addr != 0) {
        printf("Addr %x must be 0 for ELA records [%s]\n",
               addr, linePtr);
        exit(1);
      }
      sscanf(&linePtr[9],"%04x",&baseAddr);
      baseAddr <<= 16;
      if (iline==0)
        _startAddr = baseAddr;
    }
    else {
      printf("Invalid record type %u [%s]\n", recordType, linePtr);
      exit(1);
    }
    iline++;
  }

  _write_size = _read_size = _endAddr-_startAddr+1;

  //  Pad to nearest 256 Byte boundary with 0xff
  while ( _write_size & 0xff ) {
    _addrList.push_back(++_endAddr);
    _dataList.push_back(0xff);
    _write_size++;
  }

  fclose(f);
}
