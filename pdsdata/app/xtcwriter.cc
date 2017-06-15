#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/TypeId.hh"

using namespace Pds;
#define BUFSIZE 0x4000000

int main() {
  void* buf = malloc(BUFSIZE);
  bzero(buf,BUFSIZE);
  Dgram& dgram = *(Dgram*)buf;
  TypeId tid(TypeId::Id_Xtc,1);
  Xtc& xtc = *new(&dgram.xtc) Xtc(tid);
  printf("%d\n",xtc.sizeofPayload());
  return 0;
}
