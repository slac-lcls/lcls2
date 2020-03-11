
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>

#include "psdaq/tpr/Client.hh"
#include "psdaq/tpr/Module.hh"
#include "psdaq/tpr/Frame.hh"

#include <string>

using namespace Pds::Tpr;

//static bool     verbose = false;

extern int optind;

static void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("          -t <dev>  : <tpr a/b>\n");
  printf("          -c <chan> : logic channel\n");
  printf("          -o <outp> : bit mask of outputs\n");
  printf("          -d <clks> : delay\n");
  printf("          -w <clks> : width\n");
  printf("          -r <rate> : fixed rate\n");
}

int main(int argc, char** argv) {

  extern char* optarg;
  char tprid='a';

  int c;
  bool lUsage = false;
  unsigned output  = 0;
  unsigned channel = 10;
  unsigned rate    = 5;
  unsigned delay   = 0;
  unsigned width   = 1;

  //char* endptr;

  while ( (c=getopt( argc, argv, "c:d:w:o:t:r:h?")) != EOF ) {
    switch(c) {
    case 'c':
      channel = strtoul(optarg,NULL,0);
      break;
    case 'd':
      delay = strtoul(optarg,NULL,0);
      break;
    case 'w':
      width = strtoul(optarg,NULL,0);
      break;
    case 't':
      tprid  = optarg[0];
      if (strlen(optarg) != 1) {
        printf("%s: option `-r' parsing error\n", argv[0]);
        lUsage = true;
      }
      break;
    case 'o':
      output = strtoul(optarg,NULL,0);
      break;
    case 'r':
      rate = strtoul(optarg,NULL,0);
      break;
    case 'h':
      usage(argv[0]);
      exit(0);
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

  char evrdev[16];
  sprintf(evrdev,"/dev/tpr%c",tprid);

  Client client(evrdev,channel);

  for(unsigned i=0; output; i++) {
    if (output & (1<<i)) {
      client.setup(i, delay, width);
      output &= ~(1<<i);
    }
  }
  client.start(TprBase::FixedRate(rate));
  client.reg().base.dump();
  client.release();

  return 0;
}
