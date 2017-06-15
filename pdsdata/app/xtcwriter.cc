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

  // this is the datagram, which gives you an "xtc" for free
  Dgram& dgram = *(Dgram*)buf;
  TypeId tid1(TypeId::Id_Xtc,1);
  dgram.xtc.contains = TypeId(TypeId::Id_Xtc,1);

  // append an xtc
  TypeId tid2(TypeId::Id_Xtc,2);
  Xtc& xtc2 = *new(&dgram.xtc) Xtc(tid2);

  // append an xtc
  TypeId tid3(TypeId::Id_Xtc,3);
  Xtc& xtc3 = *new(&dgram.xtc) Xtc(tid3);

  return 0;
}
