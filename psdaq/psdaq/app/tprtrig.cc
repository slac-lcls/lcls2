
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
  printf("          -D <unit> : delay tap\n");
  printf("          -P <0/1>  : polarity (falling/rising)\n");
  printf("          -e <code> : event code\n");
  printf("          -r <rate> : fixed rate\n");
  printf("          -p <part> : partition\n");
  printf("          -C        : clksel (0=LCLS1,1=LCLS2[default]\n");
  printf("          -M        : modsel (0=clksel[default],1=LCLS1,2=LCLS2\n");
  printf("          -z        : wait for key before exit\n");
}

int main(int argc, char** argv) {

  extern char* optarg;
  char tprid='a';

  int c;
  bool lUsage = false;
  bool lSleep = false;
  unsigned output  = 0;
  unsigned channel = 10;
  int      mode    = -1;
  int      rate    = -1;
  unsigned delay   = 0;
  unsigned width   = 1;
  unsigned tap     = 0;
  unsigned polarity = 0; // Falling
  int  clksel      = 1;
  int  modsel      = 0;

  //char* endptr;

  while ( (c=getopt( argc, argv, "c:d:D:w:o:t:r:e:p:zC:M:P:h?")) != EOF ) {
    switch(c) {
    case 'C':
        clksel = strtoul(optarg,NULL,0);
        break;
    case 'M':
        modsel = strtoul(optarg,NULL,0);
        break;
    case 'c':
      channel = strtoul(optarg,NULL,0);
      break;
    case 'd':
      delay = strtoul(optarg,NULL,0);
      break;
    case 'D':
      tap = strtoul(optarg,NULL,0);
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
    case 'e':
      if (mode>=0) {
        printf("Only one rate selection mode (r/e/p) allowed\n");
        lUsage = true;
      }
      mode = 0;
      rate = strtoul(optarg,NULL,0);
      break;
    case 'r':
      if (mode>=0) {
        printf("Only one rate selection mode (r/e/p) allowed\n");
        lUsage = true;
      }
      mode = 1;
      rate = strtoul(optarg,NULL,0);
      break;
    case 'p':
      if (mode>=0) {
        printf("Only one rate selection mode (r/e/p) allowed\n");
        lUsage = true;
      }
      mode = 2;
      rate = strtoul(optarg,NULL,0);
      break;
    case 'P':
      polarity = strtoul(optarg,NULL,0);
      break;
    case 'z':
      lSleep = true;
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

  static const char* names[] = {"EventCode","FixedRate","Partition"};
  printf("Configuring channel %u outputs 0x%x for %s %u\n",
         channel, output, names[mode], rate);

  Client client(evrdev,channel,clksel>0,(Client::ModeSel)modsel);

  client.reg().tpr.dump();

  for(unsigned i=0; output; i++) {
    if (output & (1<<i)) {
      client.setup(i, delay, width, polarity, tap);
      output &= ~(1<<i);
    }
  }

  switch( mode ) {
  case 0: client.start(TprBase::EventCode(rate)); break;
  case 1: client.start(TprBase::FixedRate(rate)); break;
  case 2: client.start(TprBase::Partition(rate)); break;
  };

  client.reg().base.dump();
  for(unsigned i=0; i<4; i++) {
      usleep(1000000);
      client.reg().base.dump();
  }

  client.release();

  if (lSleep) {
    printf("%s: waiting for <Ctrl-C> (SIGINT)\n", argv[0]);
    while (true) {
      sleep(30);
    }
    printf("done\n");
  }

  return 0;
}
