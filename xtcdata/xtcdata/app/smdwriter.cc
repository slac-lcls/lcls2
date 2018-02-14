#include <fcntl.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

// additions from xtc writer
#include <type_traits>

#include "xtcdata/xtc/XtcFileIterator.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/VarDef.hh"

// additions from xtc writer
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TypeId.hh"

using namespace XtcData;
using std::string;

#define BUFSIZE 0x4000000

class SmdDef:public VarDef
{
public:
  enum index
    {
      intOffset,
      maxNum
    };

   SmdDef()
   {
     detVec.push_back({"intOffset", UINT64});
   }
};

void add_names(Xtc& parent, std::vector<NameIndex>& namesVec) 
{
  Alg alg("offsetAlg",0,0,0);
  Names& fexNames = *new(parent) Names("info", alg, "offset", "");
  fexNames.add_vec<SmdDef>(parent);
  namesVec.push_back(NameIndex(fexNames));
}

void usage(char* progname)
{
  fprintf(stderr, "Usage: %s -f <filename> [-h]\n", progname);
}

int main(int argc, char* argv[])
{
  /*
   * The smdwriter reads an xtc file, extracts
   * payload size for each event datagram,
   * then writes out (fseek) offset in smd.xtc file.
   */ 
  int c;
  char* xtcname = 0;
  int parseErr = 0;

  while ((c = getopt(argc, argv, "hf:")) != -1) {
    switch (c) {
      case 'h':
        usage(argv[0]);
        exit(0);
      case 'f':
        xtcname = optarg;
        break;
      default:
        parseErr++;
    }
  }

  if (!xtcname) {
    usage(argv[0]);
    exit(2);
  }

  // Read input xtc file
  int fd = open(xtcname, O_RDONLY | O_LARGEFILE);
  if (fd < 0) {
    fprintf(stderr, "Unable to open file '%s'\n", xtcname);
    exit(2);
  }

  XtcFileIterator iter(fd, BUFSIZE);
  Dgram* dgIn;

  // Prepare output smd.xtc file
  FILE* xtcFile = fopen("smd.xtc", "w");
  if (!xtcFile) {
    printf("Error opening output xtc file.\n");
    return -1;
  }

  // Setting names
  void* configbuf = malloc(BUFSIZE);
  Dgram& config = *(Dgram*)configbuf;
  TypeId tid(TypeId::Parent, 0);
  config.xtc.contains = tid;
  config.xtc.damage = 0;
  config.xtc.extent = sizeof(Xtc);
  std::vector<NameIndex> namesVec;
  add_names(config.xtc, namesVec);
  if (fwrite(&config, sizeof(config) + config.xtc.sizeofPayload(), 1, xtcFile) != 1) {
    printf("Error writing configure to output xtc file.\n");
    return -1;
  }

  // Writing out data
  void* buf = malloc(BUFSIZE);
  unsigned eventId = 0;
  unsigned nowOffset = 0;
  while ((dgIn = iter.next())) {
    Dgram& dgOut = *(Dgram*)buf;
    TypeId tid(TypeId::Parent, 0);
    dgOut.xtc.contains = tid;
    dgOut.xtc.damage = 0;
    dgOut.xtc.extent = sizeof(Xtc);

    unsigned nameId = 0; // smd only has one nameId
    CreateData smd(dgOut.xtc, namesVec[nameId], nameId);
    smd.set_value(SmdDef::intOffset, nowOffset);
    
    printf("Read evt: %4d Dgram size: %8lu Payload size: %8d Writing offset: %10d\n", 
        eventId++, sizeof(*dgIn), dgIn->xtc.sizeofPayload(), nowOffset);

    if (fwrite(&dgOut, sizeof(dgOut) + dgOut.xtc.sizeofPayload(), 1, xtcFile) != 1) {
      printf("Error writing to output xtc file.\n");
      return -1;
    }

    // Update the offset
    nowOffset += sizeof(*dgIn) + dgIn->xtc.sizeofPayload();

  }
  printf("Done.\n");
  fclose(xtcFile);
  ::close(fd);
  
  return 0;

}
