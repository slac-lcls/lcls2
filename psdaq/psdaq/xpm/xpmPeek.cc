#include <string>
#include <sstream>

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>

#include "psdaq/xpm/Module.hh"

using Pds::Xpm::Module;
using Pds::Xpm::LinkStatus;
using std::string;

extern int optind;

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -a <IP addr> (default: 10.0.2.102)\n"
         "         -p <port>    (default: 8193)\n"
         "         -P <prefix>  (default: DAQ:LAB2\n");
}

int main(int argc, char** argv)
{
  extern char* optarg;

  int c;
  bool lUsage = false;

  const char* ip = "10.0.2.102";
  unsigned short port = 8192;

  while ( (c=getopt( argc, argv, "a:p:h")) != EOF ) {
    switch(c) {
    case 'a':
      ip = optarg;
      break;
    case 'p':
      port = strtoul(optarg,NULL,0);
      break;
    case '?':
    default:
      lUsage = true;
      break;
    }
  }

  if (optind < argc) {
    printf("%s: invalid argument -- %s\n",argv[0], argv[optind]);
    lUsage = true;
  }

  if (lUsage) {
    usage(argv[0]);
    exit(1);
  }

  Pds::Cphw::Reg::set(ip, port, 0);

  Module* m = Module::locate();
  m->init();

  printf("link Tx  Rx  Xpm rxErr rxRcv\n");
  for(unsigned i=0; i<32; i++) {
    LinkStatus s = m->linkStatus(i);
    printf("%02u  %c  %c  %c  %04x  %08x\n",
           i, s.txReady?'T':'F', s.rxReady?'T':'F', s.isXpm?'T':'F', s.rxErrs, s.rxRcvs);
  }

  return 0;
}
