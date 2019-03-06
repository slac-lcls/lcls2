#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include "XtcMonitorClient.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TransitionId.hh"

using namespace XtcData;
using namespace Pds::MonReq;

class MyMonitorClient : public XtcMonitorClient {
public:
  MyMonitorClient(const char* fname)  : _fname(fname), _c(0) {}
  int processDgram(Dgram* dg) {
    bool lclose=false;
    bool lerror=false;
    switch(dg->seq.service()) {
    case TransitionId::Unconfigure:
      break;
    case TransitionId::Configure:  // cache the configuration
      if (_c) delete[] _c;
      _sizeof_c = sizeof(*dg)+dg->xtc.sizeofPayload();
      _c = new char[_sizeof_c];
      memcpy(_c,dg,_sizeof_c);
      break;
    case TransitionId::BeginRecord:   // write the configure (and beginrecord)
      _f = fopen(_fname,"w");
      if (!_f) {
        perror("Error opening output xtc file");
        lerror=true;
        break;
      }
      if (fwrite(_c, _sizeof_c, 1, _f)!=1) {
        lerror=true;
        lclose=true;
        break;
      }
    default:                            // write all other transitions
      if (fwrite(dg, sizeof(*dg)+dg->xtc.sizeofPayload(), 1, _f)!=1) {
        lerror=true;
        lclose=true;
      }
    }

    lclose |= dg->seq.service()==TransitionId::EndRecord;

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
          "[-h]\n", progname);
}


int main(int argc, char* argv[]) {
  int c;
  const char* partitionTag = 0;
  const char* fname = 0;
  unsigned index = 0;

  while ((c = getopt(argc, argv, "?hi:p:f:")) != -1) {
    switch (c) {
    case '?':
    case 'h':
      usage(argv[0]);
      exit(0);
    case 'i':
      index = strtoul(optarg,NULL,0);
      break;
    case 'p':
      partitionTag = optarg;
      break;
    case 'f':
      fname = optarg;
      break;
    default:
      usage(argv[0]);
    }
  }

  if (!fname) {
    usage(argv[0]);
    return 1;
  }

  MyMonitorClient myClient(fname);
  fprintf(stderr, "myClient returned: %d\n", myClient.run(partitionTag,index,index));

  return 1;
}
