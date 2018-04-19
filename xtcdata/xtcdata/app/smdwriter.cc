
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

class FexDef:public VarDef
{
public:
  enum index
    {
      floatFex,
      arrayFex,
      intFex
    };

  FexDef()
   {
       NameVec.push_back({"floatFex",Name::DOUBLE});
       NameVec.push_back({"arrayFex",Name::FLOAT,2});
       NameVec.push_back({"intFex",Name::INT64});
   }
} FexDef;

class PgpDef:public VarDef
{
public:
  enum index
    {
      floatPgp,
      array0Pgp,
      intPgp,
      array1Pgp
    };


   PgpDef()
   {
     NameVec.push_back({"floatPgp",Name::DOUBLE,0});
     NameVec.push_back({"array0Pgp",Name::FLOAT,2});
     NameVec.push_back({"intPgp",Name::INT64,0});
     NameVec.push_back({"array1Pgp",Name::FLOAT,2});
   }
} PgpDef;

class PadDef:public VarDef
{
public:
  enum index
    {
      arrayRaw
    };


  PadDef()
   {
     Alg segmentAlg("cspadseg",2,3,42);
     NameVec.push_back({"arrayRaw", segmentAlg});
   }
} PadDef;

class SmdDef:public VarDef
{
public:
  enum index
    {
      intOffset,
      intDgramSize
    };

   SmdDef()
   {
     NameVec.push_back({"intOffset", Name::UINT64});
     NameVec.push_back({"intDgramSize", Name::UINT64});
   }
} SmdDef;

void add_names(Xtc& parent, std::vector<NameIndex>& namesVec) 
{
    Alg hsdRawAlg("raw",0,0,0);
    Names& frontEndNames = *new(parent) Names("xpphsd", hsdRawAlg, "hsd", "detnum1234");
    frontEndNames.add(parent,PgpDef);
    namesVec.push_back(NameIndex(frontEndNames));

    Alg hsdFexAlg("fex",4,5,6);
    Names& fexNames = *new(parent) Names("xpphsd", hsdFexAlg, "hsd","detnum1234");
    fexNames.add(parent, FexDef);
    namesVec.push_back(NameIndex(fexNames));

    unsigned segment = 0;
    Alg cspadRawAlg("raw",2,3,42);
    Names& padNames = *new(parent) Names("xppcspad", cspadRawAlg, "cspad", "detnum1234", segment);
    Alg segmentAlg("cspadseg",2,3,42);
    padNames.add(parent, PadDef);
    namesVec.push_back(NameIndex(padNames)); 

    Alg alg("offsetAlg",0,0,0);
    Names& offsetNames = *new(parent) Names("info", alg, "offset", "");
    offsetNames.add(parent,SmdDef);
    namesVec.push_back(NameIndex(offsetNames));
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
  uint64_t nowOffset = 0;
  uint64_t nowDgramSize = 0;

  printf("\nStart writing offsets.\n"); 
  while ((dgIn = iter.next())) {
    Dgram& dgOut = *(Dgram*)buf;
    TypeId tid(TypeId::Parent, 0);
    dgOut.xtc.contains = tid;
    dgOut.xtc.damage = 0;
    dgOut.xtc.extent = sizeof(Xtc);

    unsigned nameId = 3; // offset is the last
    CreateData smd(dgOut.xtc, namesVec, nameId);
    smd.set_value(SmdDef::intOffset, nowOffset);
    nowDgramSize = sizeof(*dgIn) + dgIn->xtc.sizeofPayload();
    smd.set_value(SmdDef::intDgramSize, nowDgramSize);
    
    if (eventId > 0) { 
        printf("Read evt: %4d Header size: %8lu Payload size: %8d Writing offset: %10d size: %10d\n", 
        eventId, sizeof(*dgIn), dgIn->xtc.sizeofPayload(), nowOffset, nowDgramSize);

        if (fwrite(&dgOut, sizeof(dgOut) + dgOut.xtc.sizeofPayload(), 1, xtcFile) != 1) {
            printf("Error writing to output xtc file.\n");
            return -1;
        }
    } else {
        printf("Skip evt: 0 (config)\n");
    }

    // Update the offset
    nowOffset += sizeof(*dgIn) + dgIn->xtc.sizeofPayload();
    eventId++;
  }
  printf("Done.\n");
  fclose(xtcFile);
  ::close(fd);
  
  return 0;

}
