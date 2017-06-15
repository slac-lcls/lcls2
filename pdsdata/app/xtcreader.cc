#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <map>
#include <string>

#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/xtc/ProcInfo.hh"
#include "pdsdata/xtc/XtcIterator.hh"
#include "pdsdata/xtc/XtcFileIterator.hh"
static unsigned eventCount = 0;

using namespace Pds;
using std::map;
using std::string;

static map<Src,string> aliasMap;

class myLevelIter : public XtcIterator {
public:
  enum {Stop, Continue};
  myLevelIter(Xtc* xtc, unsigned depth, long long int lliOffset) : XtcIterator(xtc), _depth(depth), _lliOffset(lliOffset) {}

  int process(Xtc* xtc) {
    unsigned      i         =_depth; while (i--) printf("  ");
    Level::Type   level     = xtc->src.level();
    printf("%s level  offset %Ld (0x%Lx), payload size %d contains %s damage 0x%x: ",
           Level::name(level), _lliOffset, _lliOffset, xtc->sizeofPayload(), TypeId::name(xtc->contains.id()),
           xtc->damage.value());
    long long lliOffsetPayload = _lliOffset + sizeof(Xtc);
    _lliOffset += sizeof(Xtc) + xtc->sizeofPayload();
     
    switch (xtc->contains.id()) {
    case (TypeId::Id_Xtc) : {
      myLevelIter iter(xtc,_depth+1, lliOffsetPayload);
      iter.iterate();
      break;
    }
    default :
      printf("Unsupported TypeId %s (value = %d)\n", TypeId::name(xtc->contains.id()), (int) xtc->contains.id());
      break;
    }
    return Continue;
  }
private:
  unsigned       _depth;
  long long int  _lliOffset;

};

void usage(char* progname) {
  fprintf(stderr,"Usage: %s -f <filename> [-h]\n", progname);
}

int main(int argc, char* argv[]) {
  int c;
  char* xtcname=0;
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

  int fd = open(xtcname, O_RDONLY | O_LARGEFILE);
  if (fd < 0) {
    fprintf(stderr, "Unable to open file '%s'\n", xtcname);
    exit(2);
  }

  XtcFileIterator iter(fd,0x4000000);
  Dgram* dg;
  long long int lliOffset = lseek64( fd, 0, SEEK_CUR );  
  while ((dg = iter.next())) {
    printf("%s transition: time %d.%09d, fid/ticks 0x%0x/0x%x, env 0x%x, offset %Ld (0x%Lx), payloadSize %d\n",
           TransitionId::name(dg->seq.service()),
           dg->seq.clock().seconds(),dg->seq.clock().nanoseconds(),
           dg->seq.stamp().fiducials(),dg->seq.stamp().ticks(), 
           dg->env.value(),
           lliOffset, lliOffset, dg->xtc.sizeofPayload());
    myLevelIter iter(&(dg->xtc),0, lliOffset + sizeof(Xtc) + sizeof(*dg) - sizeof(dg->xtc));
    iter.iterate();
    lliOffset = lseek64( fd, 0, SEEK_CUR ); // get the file offset for the next iteration
  }

  ::close(fd);
  return 0;
}
