#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/TypeId.hh"
#include "pdsdata/xtc/XtcIterator.hh"

using namespace Pds;
#define BUFSIZE 0x4000000

class myLevelIter : public XtcIterator {
public:
  enum {Stop, Continue};
  myLevelIter(Xtc* xtc, unsigned depth) : XtcIterator(xtc), _depth(depth) {}

  int process(Xtc* xtc) {
    unsigned i =_depth;
    printf ("here in process\n");
    switch (xtc->contains.id()) {
    case (TypeId::Id_Xtc) : {
      printf("depth %d\n",_depth);
      myLevelIter iter(xtc,_depth+1);
      iter.iterate();
      break;
    }
    default :
      printf("TypeId %s (value = %d)\n", TypeId::name(xtc->contains.id()), (int) xtc->contains.id());
      break;
    }
    return Continue;
  }
private:
  unsigned       _depth;
  long long int  _lliOffset;

};


int main() {
  void* buf = malloc(BUFSIZE);
  bzero(buf,BUFSIZE);

  // this is the datagram, which gives you an "xtc" for free
  Dgram& dgram = *(Dgram*)buf;
  TypeId tid(TypeId::Id_Xtc,0);
  dgram.xtc.contains = tid;
  dgram.xtc.extent = sizeof(Xtc);
  printf("1: %d\n",dgram.xtc.extent);

  // make a child xtc.  this one will become our camera data
  TypeId tid_child1(TypeId::Id_EpixConfig,1);
  Xtc& xtc2 = *new(&dgram.xtc) Xtc(tid_child1);
  printf("2: %d\n",dgram.xtc.extent);

  // make a child xtc.  this one will become our descriptor
  TypeId tid_child2(TypeId::Id_Frame,1);
  Xtc& xtc3 = *new(&dgram.xtc) Xtc(tid_child2);
  printf("3: %d\n",dgram.xtc.extent);

  myLevelIter iter(&dgram.xtc,0);
  iter.iterate();
  return 0;
}
