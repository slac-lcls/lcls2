#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include "ShmemClient.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TransitionId.hh"

using namespace XtcData;
using namespace psalg::shmem;

class MyShmemClient : public ShmemClient {
public:
  MyShmemClient(int rate) : _rate(rate) {}
private:
  int _rate;
};

void usage(char* progname) {
  fprintf(stderr,"Usage: %s "
          "[-p <partitionTag>] "
          "[-i <index>] "
          "[-r <rate>] "
          "[-h]\n", progname);
}


int main(int argc, char* argv[]) {
  int c;
  const char* partitionTag = 0;
  unsigned index = 0;
  int rate = 0;

  while ((c = getopt(argc, argv, "?hi:p:r:")) != -1) {
    switch (c) {
    case '?':
    case 'h':
      usage(argv[0]);
      exit(0);
    case 'i':
      index = strtoul(optarg,NULL,0);
      break;
    case 'r':
      rate = strtoul(optarg,NULL,0);
      break;
    case 'p':
      partitionTag = optarg;
      break;
    default:
      usage(argv[0]);
    }
  }

  MyShmemClient myClient(rate);
  myClient.connect(partitionTag,index);
  while(1)
    {
    int ev_index,buf_size;
    void *dgram = myClient.get(ev_index,buf_size);
    if(!dgram) break;
    printf("ShmemClient dgram %p index %d size %d\n",dgram,ev_index,buf_size);
    myClient.free(ev_index,buf_size);
    }

  return 1;
}
