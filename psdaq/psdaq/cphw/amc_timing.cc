#include "psdaq/cphw/AmcTiming.hh"

#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

#include <new>
#include <vector>

using namespace Pds::Cphw;

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -a <ip address, dotted notation>\n");
  printf("         -2 With -r or -R, select LCLS-II timing (default)\n");
  printf("         -i With -r or -R, invert polarity (default is not)\n");
  printf("         -r <source> Reset timing to (0=RTM0, 1=FPGA, 2=BP, 3=RTM1)\n");
  printf("         -t Align target\n");
  printf("         -x Value to override XBar output mapping with\n");
  printf("         -X <out,inp> individual XBar channel mapping\n");
  printf("         -m Measure and dump clocks\n");
  printf("         -c Clear counters\n");
  printf("         -d dump timing\n");
  printf("         -D dump timing sparse\n");
}

int main(int argc, char* argv[])
{
  extern char* optarg;
  char c;
  char* endptr;

  const char* ip = "192.168.2.10";
  unsigned short port = 8192;
  bool lcls2 = true;
  bool lReset = false;
  unsigned iSource = 2;
  bool invertPolarity = false;
  bool measureClks = false;
  bool lClear = false;
  int alignTarget = -1;
  std::vector<unsigned> ovXbar;
  std::vector<unsigned> ivXbar;
  bool lDump0=false, lDump1=false;

  while ( (c=getopt( argc, argv, "a:x:X:2dDimcr:t:h")) != EOF ) {
    switch(c) {
    case 'a':
      ip = optarg;
      break;
    case '2':
      lcls2 = true;
      break;
    case 'i':
      invertPolarity = true;
      break;
    case 'm':
      measureClks = true;
      break;
    case 'c':
      lClear = true;
      break;
    case 'r':
      lReset = true;
      iSource = strtoul(optarg,NULL,0);
      break;
    case 't':
      alignTarget = strtoul(optarg,NULL,0);
      break;
    case 'x':
      { unsigned iv = strtoul(optarg,NULL,0);
        for(unsigned i=0; i<4; i++) {
          ovXbar.push_back(i);
          ivXbar.push_back(iv);
        }
      } break;
    case 'X':
      ovXbar.push_back(strtoul(optarg,&endptr,0));
      ivXbar.push_back(strtoul(endptr+1,NULL,0));
      break;
    case 'd':
      lDump0 = true;
      break;
    case 'D':
      lDump1 = true;
      break;
    default:
      usage(argv[0]);
      return 0;
    }
  }

  Pds::Cphw::Reg::set(ip, port, 0);

  Pds::Cphw::AmcTiming* t = new((void*)0) Pds::Cphw::AmcTiming;
  printf("buildStamp %s\n",t->version.buildStamp().c_str());

  if (lDump0) {
    t->ring0.enable(false);
    t->ring0.clear();
    t->ring0.enable(true);
    usleep(10);
    t->ring0.enable(false);
    t->ring0.dump();
  }

  if (lDump1) {
    t->ring1.enable(false);
    t->ring1.clear();
    t->ring1.enable(true);
    usleep(10);
    t->ring1.enable(false);
    t->ring1.dump();
  }

  if (alignTarget >= 0) {
    t->setRxAlignTarget(alignTarget);
    t->bbReset();
  }

  if (lReset) {
    t->xbar.setOut( XBar::FPGA, (XBar::Map)iSource );
    if (!lcls2)
      t->setLCLS();
    else
      t->setLCLSII();
    t->setPolarity(invertPolarity);
    t->bbReset();
    usleep(1000);
    t->resetStats();
    usleep(1000);
  }

  for(unsigned i=0; i<ovXbar.size(); i++) 
    t->xbar.outMap[ovXbar[i]]=ivXbar[i];

  t->xbar.dump();

  t->dumpStats();

  if (measureClks)
  {
    unsigned rxT0 = unsigned(t->rx.RxRecClks);
    unsigned txT0 = unsigned(t->rx.TxRefClks);
    usleep(1000000);
    unsigned rxT1 = unsigned(t->rx.RxRecClks);
    unsigned txT1 = unsigned(t->rx.TxRefClks);
    unsigned dRxT = (rxT1 - rxT0) << 4;
    unsigned dTxT = (txT1 - txT0) << 4;
    printf("\n");
    printf("%10.10s: 0x%08x = %u\n","1S dT(rx)",dRxT, dRxT);
    printf("%10.10s: 0x%08x = %u\n","1S dT(tx)",dTxT, dTxT);
  }

  //  t->dumpRxAlign();

  if (lClear)
    t->resetStats();

  sleep(1);

  return 0;
}
