#include "pdsdata/xtc/Descriptor.hh"
#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/TypeId.hh"
#include "pdsdata/xtc/XtcIterator.hh"

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>

using namespace Pds;
#define BUFSIZE 0x4000000
#define NDGRAM 2

class MyData : public Data
{
public:
  void* operator new(size_t size, void* p)
  {
    return p;
  }
  MyData(float f, int i1, int i2) : Data(sizeof(*this))
  {
    _fdata = f;
    _idata = i1;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        array[i][j] = i * j;
      }
    }
  }

private:
  float _fdata;
  float array[3][3];
  int _idata;
};

void pgpXtcExample(Xtc* parent, char* intName, char* floatName, char* arrayName, int vals[3])
{
  // make a child xtc with detector data and descriptor
  TypeId tid_child(TypeId::Data, 0);
  Xtc& xtcChild = *new (parent->next()) Xtc(tid_child);

  // creation of fixed-length data in xtc
  MyData& d = *new (xtcChild.alloc(sizeof(MyData))) MyData(vals[0],vals[1],vals[2]);

  // creation of variable-length data in xtc
  DescriptorManager descMgr(xtcChild.next());
  descMgr.add(floatName, FLOAT);

  int shape[] = {3, 3};
  descMgr.add(arrayName, FLOAT, 2, shape);

  descMgr.add(intName, INT32);

  xtcChild.alloc(descMgr.size());

  // update parent xtc with our new size.
  parent->alloc(xtcChild.extent);

}

void fexXtcExample(Xtc* parent, char* intName, char* floatName, char* arrayName, int vals[3])
{
  TypeId tid_child(TypeId::Data, 0);
  Xtc& xtcChild = *new (parent->next()) Xtc(tid_child);

  Data& d = *new (xtcChild.alloc(sizeof(Data))) Data();
  Descriptor& desc = d.desc();

  // it is necessary to fill the descriptor completely before adding data
  DescriptorManager descMgr(&desc);
  descMgr.add("fexfloat1", FLOAT);
  descMgr.add("fexint1", INT32);

  d.set_value("fexfloat1",5.0);
  d.set_value("fexint1",2);

  // fix these next lines
  xtcChild.alloc(descMgr.size());
  // update parent xtc with our new size.
  parent->alloc(xtcChild.extent);

}


int main() {
  void* buf = malloc(BUFSIZE);
  int vals1[3];
  int vals2[3];
  FILE* xtcFile = fopen("data.xtc","w");
  if(!xtcFile){
    printf("Error opening output xtc file.\n");
    return -1;
  }

  for (int i=0; i<NDGRAM; i++) {
    Dgram& dgram = *(Dgram*)buf;
    TypeId tid(TypeId::Parent, 0);
    dgram.xtc.contains = tid;
    dgram.xtc.damage = 0;
    dgram.xtc.extent = sizeof(Xtc);

    for (int j=0; j<3; j++) {
      vals1[j] = i+j;
      vals2[j] = 1000+i+j;
    }
    char intname[10], floatname[10], arrayname[10];
    sprintf(intname,"%s%d","int",i);
    sprintf(floatname,"%s%d","float",i);
    sprintf(arrayname,"%s%d","array",i);
    pgpXtcExample(&dgram.xtc, intname, floatname, arrayname, vals1);
    sprintf(intname,"%s%d","int",i+1);
    sprintf(floatname,"%s%d","float",i+1);
    sprintf(arrayname,"%s%d","array",i+1);
    pgpXtcExample(&dgram.xtc, intname, floatname, arrayname, vals2);

    if(fwrite(&dgram, sizeof(dgram)+dgram.xtc.sizeofPayload(), 1, xtcFile)!=1){
      printf("Error writing to output xtc file.\n");
      return -1;
    }
  }
  fclose(xtcFile);

  return 0;
}
