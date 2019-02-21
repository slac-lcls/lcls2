
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <arpa/inet.h>
#include <poll.h>
#include <signal.h>
#include <new>

#include "psdaq/hsd/Module.hh"
#include "psdaq/hsd/Globals.hh"
#include "psdaq/hsd/FlashController.hh"
#include "psdaq/mmhw/AxiVersion.hh"
#include "psdaq/mmhw/McsFile.hh"
#include "psdaq/mmhw/HexFile.hh"

#include <string>

extern int optind;

using namespace Pds::HSD;
using Pds::Mmhw::AxiVersion;

static void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -d <dev id>\n");
  printf("         -R filename <read new PROM>\n");
  printf("         -r words to read\n");
  printf("         -V filename <verify against PROM>\n");
  printf("         -W filename <write new PROM>\n");
  printf("         -C filename <convert file to text (/tmp/hsd_flash.cvt)>\n");
  printf("         -s <use slow method/no fifo>\n");
  printf("         -v (verbose)\n");
}

static void write_file(const char* fname, const std::vector<uint8_t>& v);

int main(int argc, char** argv) {

  extern char* optarg;

  const char* devName = "/dev/qadca";
  int c;
  bool lUsage = false;

  unsigned    nRead  =16*1024;
  const char* fRead  =0;
  const char* fVerify=0;
  const char* fWrite =0;
  const char* fCvtIn  =0;
  const char* fCvtOut ="/tmp/hsd_flash.cvt";

  while ( (c=getopt( argc, argv, "d:r:sC:R:V:W:v")) != EOF ) {
    switch(c) {
    case 'd': devName = optarg; break;
    case 'r': nRead   = strtoul(optarg,NULL,0); break;
    case 's': FlashController::useFifo(false); break;
    case 'C': fCvtIn  = optarg; break;
    case 'R': fRead   = optarg; break;
    case 'V': fVerify = optarg; break;
    case 'W': fWrite  = optarg; break;
    case 'v': FlashController::verbose(true); break;
    default:  lUsage = true;    break;
    }
  }

  if (lUsage) {
    usage(argv[0]);
    exit(1);
  }

  int fd = open(devName, O_RDWR);
  if (fd<0) {
    perror("Open device failed");
    return -1;
  }

  Module* p = Module::create(fd);

  AxiVersion& vsn = *reinterpret_cast<AxiVersion*>(p->reg());
  printf("buildStamp: %s\n", vsn.buildStamp().c_str());

  while (fCvtIn) {
    const char* extp = strrchr(fCvtIn,'.');
    if (!extp) {
      printf("No file extension on %s\n",fCvtIn);
      break;
    }

    std::vector<uint8_t> v;
    if (strcmp(extp,".hex")==0) {
      Pds::Mmhw::HexFile f(fCvtIn);
      write_file(fCvtOut,f.data());
    }
    else if (strcmp(extp,".mcs")==0) {
      Pds::Mmhw::McsFile m(fCvtIn);
      Pds::Mmhw::HexFile f(m);
      write_file(fCvtOut,f.data());
    }
    break;
  }

  FlashController& flash = p->flash();

  if (fRead)
    write_file(fRead,flash.read(nRead));

  if (fVerify)
    flash.verify(fVerify);

  if (fWrite)
    flash.write(fWrite);

  return 0;
}

void write_file(const char* fname, const std::vector<uint8_t>& v)
{
  FILE* f = fopen(fname,"w");

  unsigned w = 0;
  for(unsigned i=0; i<v.size(); i++) {
    w >>= 4;
    w |= unsigned(v[i])<<24;
    if ((i&3)==3) {
      fprintf(f,"%08x%c",w,(i&0x1f)==0x1f ? '\n':' ');
    }
  }

  if ((v.size()&0x1f)!=0) {
    for(unsigned i=0; i<((v.size()&3)^3); i++)
      w >>= 4;
    for(unsigned i=0; i<((v.size()>>2)^0x7); i++) {
      fprintf(f,"%08x ",w);
      w=0;
    }
    fprintf(f,"\n");
  }

  fclose(f);
}

