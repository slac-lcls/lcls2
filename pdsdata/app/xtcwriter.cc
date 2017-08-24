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
#define NDGRAM 1

class PgpData
{
public:
  void* operator new(size_t size, void* p) {return p;}
  PgpData(float f, int i1, int i2) {
    _fdata = f;
    _idata = i1;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        array[i][j] = i * j;
      }
    }
  }

  float _fdata;
  float array[3][3];
  int _idata;
};

void pgpExample(Xtc* parent, char* intName, char* floatName, char* arrayName, int vals[3])
{
  // make a child xtc with detector data and descriptor
  TypeId tid_child(TypeId::DescData, 0);
  Xtc& xtcChild = *new(parent->next()) Xtc(tid_child);

  Data& data = *new(xtcChild.payload()) Data();

  // simulates PGP data arriving, and shows the address that should be given to PGP driver
  new(data.payload()) PgpData(vals[0],vals[1],vals[2]);

  // now that data has arrived can update with the number of bytes received
  data.extend(sizeof(PgpData));

  // now fill in the descriptor
  Descriptor& desc = *new(data.next()) Descriptor();

  uint32_t offset;
  desc.add(floatName, FLOAT, offset);

  int shape[] = {3, 3};
  desc.add(arrayName, FLOAT, 2, shape, offset);
  desc.add(intName, INT32, offset);

  // update our xtc with the size of the data we have added
  xtcChild.alloc(desc.size()+data.size());

  // update parent xtc with our new size.
  parent->alloc(xtcChild.extent);

}

void fexExample(Xtc* parent)
{
  // make a child xtc with detector data and descriptor
  TypeId tid_child(TypeId::DescData, 0);
  Xtc& xtcChild = *new(parent->next()) Xtc(tid_child);

  DescData& descdata = *new(xtcChild.payload()) DescData();
  Descriptor& desc = descdata.desc();

  uint32_t offset;
  desc.add("fexfloat1", FLOAT, offset);
  desc.add("fexint1", INT32, offset);

  Data& data = *new(desc.next()) Data();
  descdata.set_value("fexfloat1",41.0);
  descdata.set_value("fexint1",42);

  printf("*** %d %d\n",desc.size(),data.size());
  // update our xtc with the size of the data we have added
  xtcChild.alloc(desc.size()+data.size());

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
    pgpExample(&dgram.xtc, intname, floatname, arrayname, vals1);
    sprintf(intname,"%s%d","int",i+1);
    sprintf(floatname,"%s%d","float",i+1);
    sprintf(arrayname,"%s%d","array",i+1);
    pgpExample(&dgram.xtc, intname, floatname, arrayname, vals2);
    fexExample(&dgram.xtc);

    if(fwrite(&dgram, sizeof(dgram)+dgram.xtc.sizeofPayload(), 1, xtcFile)!=1){
      printf("Error writing to output xtc file.\n");
      return -1;
    }
  }
  fclose(xtcFile);

  return 0;
}
