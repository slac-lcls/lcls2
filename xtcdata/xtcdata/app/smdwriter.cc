
#include <fcntl.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <iostream>
#include <sys/time.h>
#include <fstream>
#include <array>

// additions from xtc writer
#include <type_traits>

#include "xtcdata/xtc/XtcFileIterator.hh"
#include "xtcdata/xtc/VarDef.hh"

// additions from xtc writer
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TypeId.hh"

using namespace XtcData;
using std::string;
using namespace std;

#define BUFSIZE 0x4000000

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

void addNames(Xtc& parent, NamesLookup& namesLookup, unsigned nodeId)
{
    Alg alg("offsetAlg",0,0,0);
    NamesId namesId(nodeId,0);
    Names& offsetNames = *new(parent) Names("info", alg, "offset", "", namesId);
    offsetNames.add(parent,SmdDef);
    namesLookup[namesId] = NameIndex(offsetNames);
}

void usage(char* progname)
{
  fprintf(stderr, "Usage: %s -f <filename> [-h]\n", progname);
}

#define MAX_FNAME_LEN 256

int main(int argc, char* argv[])
{
  /*
   * The smdwriter reads an xtc file, extracts
   * payload size for each event datagram,
   * then writes out (fseek) offset in smd.xtc2 file.
   */ 
  int c;
  int writeTs = 0;
  char* tsname = 0;
  char* xtcname = 0;
  int parseErr = 0;
  size_t n_events = 0;
  char outname[MAX_FNAME_LEN];
  strncpy(outname, "smd.xtc2", MAX_FNAME_LEN);

  while ((c = getopt(argc, argv, "ht:n:f:o:")) != -1) {
    switch (c) {
      case 'h':
        usage(argv[0]);
        exit(0);
      case 't':
        writeTs = 1;
        tsname = optarg;
        break;
      case 'n':
        n_events = stoi(optarg);
        break;
      case 'f':
        xtcname = optarg;
        break;
      case 'o':
        strncpy(outname, optarg, MAX_FNAME_LEN);
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
  int fd = open(xtcname, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "Unable to open file '%s'\n", xtcname);
    exit(2);
  }

  XtcFileIterator iter(fd, BUFSIZE);
  Dgram* dgIn;

  // Prepare output smd.xtc2 file
  FILE* xtcFile = fopen(outname, "w");
  if (!xtcFile) {
    printf("Error opening output xtc file.\n");
    return -1;
  }
  
  // Read timestamp
  array<time_t, 500> sec_arr = {};
  array<long, 500> nsec_arr = {};
  sec_arr[0] = 0;
  nsec_arr[0] = 0;
  if (tsname != 0) {
    ifstream tsfile(tsname);
    time_t sec = 0;
    long nsec = 0;
    int i = 1;
    while (tsfile >> sec >> nsec) {
        sec_arr[i] = sec;
        nsec_arr[i] = nsec;
        cout << sec_arr[i] << " " << nsec_arr[i] << endl;
        i++;
    }  
    printf("found %d timestamps", i); 
  }
  
  // Setting names
  void* configbuf = malloc(BUFSIZE);
  Dgram& config = *(Dgram*)configbuf;
  TypeId tid(TypeId::Parent, 0);
  config.xtc.contains = tid;
  config.xtc.damage = 0;
  config.xtc.extent = sizeof(Xtc);
  NamesLookup namesLookup;
  unsigned nodeId=0;
  addNames(config.xtc, namesLookup, nodeId);
  if (fwrite(&config, sizeof(config) + config.xtc.sizeofPayload(), 1, xtcFile) != 1) {
    printf("Error writing configure to output xtc file.\n");
    return -1;
  }

  // Writing out data
  void* buf = malloc(BUFSIZE);
  unsigned eventId = 0;
  uint64_t nowOffset = 0;
  uint64_t nowDgramSize = 0;
  struct timeval tv;
  uint64_t pulseId = 0;

  printf("\nStart writing offsets.\n"); 
  if (n_events > 0) {
  }

  while ((dgIn = iter.next())) {
    Dgram& dgOut = *(Dgram*)buf;
    TypeId tid(TypeId::Parent, 0);
    dgOut.xtc.contains = tid;
    dgOut.xtc.damage = 0;
    dgOut.xtc.extent = sizeof(Xtc);

    if (writeTs != 0) {
        if (tsname != 0) {
            dgOut.seq = Sequence(TimeStamp(sec_arr[eventId], nsec_arr[eventId]), PulseId(pulseId));
        } else {
            gettimeofday(&tv, NULL);
            dgOut.seq = Sequence(TimeStamp(tv.tv_sec, tv.tv_usec), PulseId(pulseId));
            cout << tv.tv_sec << " " << tv.tv_usec << endl;
        }
    } else {
        dgOut.seq = dgIn->seq;
    }

    NamesId namesId(nodeId,0);
    CreateData smd(dgOut.xtc, namesLookup, namesId);
    smd.set_value(SmdDef::intOffset, nowOffset);
    nowDgramSize = (uint64_t)(sizeof(*dgIn) + dgIn->xtc.sizeofPayload());
    smd.set_value(SmdDef::intDgramSize, nowDgramSize);
    
    if (eventId > 0 ) {
        if (nowOffset < 0) {
            cout << "Error offset value (offset=" << nowOffset << ")" << endl;
            return -1;
        }
        if (nowDgramSize <= 0) {
            cout << "Error size value (size=" << nowDgramSize << ")" << endl;
            return -1;
        }
        if (fwrite(&dgOut, sizeof(dgOut) + dgOut.xtc.sizeofPayload(), 1, xtcFile) != 1) {
            printf("Error writing to output xtc file.\n");
            return -1;
        }
    }

    // Update the offset
    nowOffset += (uint64_t)(sizeof(*dgIn) + dgIn->xtc.sizeofPayload());
    eventId++;

    if (n_events > 0) {
        if (eventId - 1 >= n_events) {
            cout << "Stop writing. The option -n (no. of events) was set to " << n_events << endl;
            break;
        }
    }

  }// end while((dgIn...

  cout << "Finished writing smd for " << eventId - 1 << " events. Big data file size (B): " << nowOffset << endl;
  fclose(xtcFile);
  ::close(fd);
  
  return 0;

}
