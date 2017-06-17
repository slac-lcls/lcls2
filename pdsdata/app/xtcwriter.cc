#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include <unistd.h>
#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/TypeId.hh"
#include "pdsdata/xtc/Descriptor.hh"
#include "pdsdata/xtc/XtcIterator.hh"

using namespace Pds;
#define BUFSIZE 0x4000000

// generic data class that holds the size of the variable-length data and
// the location of the descriptor
class Data {
public:
  Data(uint32_t size) : _sizeofData(size) {}
  //protected:
  Descriptor& desc() {
    return *(Descriptor*)(((char*)(this))+_sizeofData);
  }
private:
  uint32_t _sizeofData;
};

// this represent the "analysis" code 
class myLevelIter : public XtcIterator {
public:
  enum {Stop, Continue};
  myLevelIter(Xtc* xtc, unsigned depth) : XtcIterator(xtc), _depth(depth) {}

  int process(Xtc* xtc) {
    unsigned i =_depth;
    switch (xtc->contains.id()) {
    case (TypeId::Parent) : {
      myLevelIter iter(xtc,_depth+1);
      iter.iterate();
      break;
    }
    case (TypeId::Data) : {
      Data& d = *(Data*)xtc->payload();
      Descriptor& desc = d.desc();
      printf("Found fields named:\n");
      for (int i=0; i<desc.num_fields; i++) {
        printf("%s\n",desc.get(i).name);
      }
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

// everything below here is inletWire code

class MyData : public Data {
public:
  void* operator new(size_t size, void* p) { return p; }
  MyData(float f, int i1, int i2) : Data(sizeof(*this)) {
    _fdata = f; _idata=i1;
  }
private:
  float _fdata;
  int   _idata;
};

int main() {
  // this is the datagram, which gives you an "xtc" for free
  Dgram& dgram = *(Dgram*)malloc(BUFSIZE);
  TypeId tid(TypeId::Parent,0);
  dgram.xtc.contains = tid;
  dgram.xtc.extent = sizeof(Xtc);

  // make a child xtc with detector data and descriptor
  TypeId tid_child(TypeId::Data,0);
  Xtc& xtcChild = *new(&dgram.xtc) Xtc(tid_child);

  // creation of fixed-length data in xtc
  MyData& d = *new(xtcChild.alloc(sizeof(MyData))) MyData(1,2,3);

  // creation of variable-length data in xtc
  DescriptorManager descMgr(xtcChild.next());
  descMgr.add("myfloat",FLOAT);
  descMgr.add("myint",INT);
  xtcChild.alloc(descMgr.size());

  // update parent xtc with our new size.
  dgram.xtc.alloc(xtcChild.sizeofPayload());

  myLevelIter iter(&dgram.xtc,0);
  iter.iterate();
  return 0;
}
