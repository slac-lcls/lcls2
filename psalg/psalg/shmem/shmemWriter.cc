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
  MyShmemClient(const char* fname) : _fname(fname), _c(0) {}
  int processDgram(Dgram* dg) {
    bool lclose=false;
    bool lerror=false;
    switch(dg->seq.service()) {
    case TransitionId::Unconfigure:
      break;
    case TransitionId::Configure:  // cache the configuration
      _f = fopen(_fname,"w");
      if (!_f) {
        perror("Error opening output xtc file");
        lerror=true;
        break;
      printf("Opened %s for writing\n",_fname);  
      }
      if (_c) delete[] _c;
      _sizeof_c = sizeof(*dg)+dg->xtc.sizeofPayload();
      _c = new char[_sizeof_c];
      memcpy(_c,dg,_sizeof_c);
      if (fwrite(_c, _sizeof_c, 1, _f)!=1) {
        lerror=true;
        lclose=true;
        break;
      }
      break;
    default:  // write all other transitions
      if (fwrite(dg, sizeof(*dg)+dg->xtc.sizeofPayload(), 1, _f)!=1) {
        lerror=true;
        lclose=true;
      }
    }

    lclose |= dg->seq.service()==TransitionId::Unconfigure;

    if (lclose)
      fclose(_f);

    return lerror ? 1:0;
  }

private:
  const char* _fname;
  FILE* _f;
  char* _c;
  unsigned _sizeof_c;
};

void usage(char* progname) {
  fprintf(stderr,"Usage: %s "
           "-f <filename> "
          "[-p <partitionTag>] "
          "[-i <index>] "
          "[-v] "
          "[-h]\n", progname);
}


int main(int argc, char* argv[]) {
  int c;
  const char* partitionTag = 0;
  const char* fname = 0;
  unsigned index = 0;
  int rate = 0;
  bool verbose = false;

  while ((c = getopt(argc, argv, "?hvi:p:f:")) != -1) {
    switch (c) {
    case '?':
    case 'h':
      usage(argv[0]);
      exit(0);
    case 'i':
      index = strtoul(optarg,NULL,0);
      break;
    case 'f':
      fname = optarg;
      break;
    case 'p':
      partitionTag = optarg;
      break;
    case 'v':
      verbose = true;
      break;
    default:
      usage(argv[0]);
    }
  }

  if (!fname) {
    usage(argv[0]);
    return 1;
  }

  MyShmemClient myClient(fname);
  myClient.connect(partitionTag,index);
  while(1)
    {
    int ev_index,buf_size;
    Dgram *dgram = (Dgram*)myClient.get(ev_index,buf_size);
    if(!dgram) break;
    if(verbose)
      printf("shmemWriter dgram trId %d index %d size %d\n",dgram->seq.service(),ev_index,buf_size);
    myClient.processDgram(dgram);
    myClient.free(ev_index,buf_size);
    }

  return 1;
}
