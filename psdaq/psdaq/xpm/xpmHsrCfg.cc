#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <arpa/inet.h>
#include <signal.h>
#include <new>

#include "psdaq/xpm/Module.hh"
#include "psdaq/cphw/AmcTiming.hh"
#include "psdaq/cphw/HsRepeater.hh"

enum { NLinks = 14 };            // Total number of links for both module types

static const unsigned hsrMap[] = { 0, 0, 0, 0, 1, 1, 1,      // AMC0
                                   3, 3, 3, 3, 4, 4, 4 };    // AMC2
static const unsigned chnMap[] = { 3, 2, 1, 0, 3, 2, 1,      // AMC0
                                   3, 2, 1, 0, 3, 2, 1 };    // AMC2

extern int optind;

using namespace Pds;
using namespace Pds::Cphw;


unsigned xpmMeasFn(void* arg)
{
  Cphw::Reg* rxErrs = (Cphw::Reg*)arg;

  unsigned e0 = *rxErrs;
  usleep(350);            // In 350 uS, the 186 MHz clock will count ~64K times
  unsigned e1 = *rxErrs;
  return (((e1 & 0x0000ffff) - (e0 & 0x0000ffff)) + 0x10000) & 0x0000ffff;
}


void usage(const char* p)
{
  printf("Usage: %s [options]\n",p);
  printf("Options: -a <IP addr (dotted notation)> : Use network <IP>\n");
  printf("         -p <port>                      : Use network <port>\n");
  printf("         -L <ds link mask>              : Enable selected link(s)\n");
  printf("         -D <DEM value>                 : DEM value for selected link(s)\n");
  printf("         -V <VOD value>                 : VOD value for selected link(s)\n");
  printf("         -E <EQ value>                  : EQ value for selected link(s)\n");
  printf("         -B                             : Apply -E, -D and/or -V to B channel(s)\n");
  printf("         -A                             : Apply -E, -D and/or -V to A channel(s)\n");
  printf("         -R                             : Reset selected link(s)\n");
  printf("         -r                             : Reset HSR registers of selected link(s)\n");
  printf("         -S                             : Reset SMBus Master of selected link(s)\n");
  printf("         -s                             : Scan EQ values of selected link(s) (overrides -E)\n");
  printf("         -d                             : Dump registers of selected link(s)\n");
}


int main(int argc, char** argv)
{
  extern char* optarg;

  int c;
  bool lUsage = false;

  const char* ip       = "10.0.2.102";
  unsigned short port  = 8192;
  unsigned linkEnables = (1 << NLinks) - 1;
  bool chB  = false;
  bool chA  = false;
  int  eqv  = -1;
  int  dem  = -1;
  int  vod  = -1;
  bool rstLinks = false;
  bool rstRegs  = false;
  bool rstSMBus = false;
  bool scan = false;
  bool dump = false;

  while ( (c=getopt( argc, argv, "a:p:L:D:V:E:BARrSsdh")) != EOF ) {
    switch(c) {
    case 'a':
      ip = optarg; break;
      break;
    case 'p':
      port = strtoul(optarg,NULL,0);
      break;
    case 'L':
      linkEnables = strtoul(optarg,NULL,0) & ((1 << NLinks) - 1);
      break;
    case 'D':
      dem = strtoul(optarg,NULL,0);
      break;
    case 'V':
      vod = strtoul(optarg,NULL,0);
      break;
    case 'E':
      eqv = strtoul(optarg,NULL,0);
      break;
    case 'B':
      chB = true;
      break;
    case 'A':
      chA = true;
      break;
    case 'R':
      rstLinks = true;
      break;
    case 'r':
      rstRegs = true;
      break;
    case 'S':
      rstSMBus = true;
      break;
    case 's':
      scan = true;
      break;
    case 'd':
      dump = true;
      break;
    case '?':
    default:
      lUsage = true;
      break;
    }
  }

  if (lUsage) {
    usage(argv[0]);
    exit(1);
  }

  Pds::Cphw::Reg::set(ip, port, 0);

  Xpm::Module* xpm = Xpm::Module::locate();

  printf("buildStamp %s\n",xpm->_version.buildStamp().c_str());
  if ( !strcasestr(xpm->_version.buildStamp().c_str(), "XPM"))
    printf("*** WARNING *** Module does not appear to be an XPM!\n");

  Pds::Cphw::TimingRx* t = &xpm->_usTiming;
  t->resetStats();
  usleep(1000);
  unsigned rxDecErrs = (unsigned)t->RxDecErrs;
  unsigned rxDspErrs = (unsigned)t->RxDspErrs;
  usleep(1000);

  if ((((unsigned)t->CSR & 0x02) == 0)      ||
      ((unsigned)t->RxDecErrs != rxDecErrs) ||
      ((unsigned)t->RxDspErrs != rxDspErrs))
  {
    printf("Exiting because source timing does not appear to be stable:\n");
    t->dumpStats();
    exit(1);
  }

  xpm->init();

  HsRepeater*  hsr = xpm->_hsRepeater;

  if (rstLinks)
  {
    unsigned links = linkEnables;
    for(unsigned i = 0; links; ++i)
    {
      if (links & (1<<i))
      {
        xpm->txLinkReset(i);
        xpm->rxLinkReset(i);
        usleep(10000);
        links &= ~(1<<i);
      }
    }
  }

  if (dem >= 0)
  {
    unsigned links = linkEnables;
    for (unsigned i = 0; links; ++i)
    {
      if (links & (1<<i))
      {
        hsr[hsrMap[i]].smbusEnable(true);
        if (chB)
          hsr[hsrMap[i]]._chB[chnMap[i]].demCtl(dem);
        if (chA)
          hsr[hsrMap[i]]._chA[chnMap[i]].demCtl(dem);
        hsr[hsrMap[i]].smbusEnable(false);
        links &= ~(1<<i);
      }
    }
  }

  if (vod >= 0)
  {
    unsigned links = linkEnables;
    for (unsigned i = 0; links; ++i)
    {
      if (links & (1<<i))
      {
        hsr[hsrMap[i]].smbusEnable(true);
        if (chB)
          hsr[hsrMap[i]]._chB[chnMap[i]].vodCtl(vod);
        if (chA)
          hsr[hsrMap[i]]._chA[chnMap[i]].vodCtl(vod);
        hsr[hsrMap[i]].smbusEnable(false);
        links &= ~(1<<i);
      }
    }
  }

  if (eqv >= 0)
  {
    unsigned links = linkEnables;
    for (unsigned i = 0; links; ++i)
    {
      if (links & (1<<i))
      {
        hsr[hsrMap[i]].smbusEnable(true);
        if (chB)
          hsr[hsrMap[i]]._chB[chnMap[i]]._eqCtl = eqv & 0xff;
        if (chA)
          hsr[hsrMap[i]]._chA[chnMap[i]]._eqCtl = eqv & 0xff;
        hsr[hsrMap[i]].smbusEnable(false);
        links &= ~(1<<i);
      }
    }
  }

  if (rstRegs)
  {
    unsigned links = linkEnables;
    for (unsigned i = 0; links; ++i)
    {
      if (links & (1<<i))
      {
        hsr[hsrMap[i]].resetRegs();
        links &= ~(1<<i);
      }
    }
  }

  if (rstSMBus)
  {
    unsigned links = linkEnables;
    for (unsigned i = 0; links; ++i)
    {
      if (links & (1<<i))
      {
        hsr[hsrMap[i]].resetSmbus();
        links &= ~(1<<i);
      }
    }
  }

  if (scan)
  {
    xpm->setL0Enabled(false);
    xpm->clearLinks();
    unsigned links = linkEnables;
    for (unsigned i = 0; links; ++i)
    {
      if (links & (1<<i))
      {
        xpm->setLink(i);

        printf("XPM link %2d: ", i);     // No \n on purpose
        unsigned idx = hsr[hsrMap[i]].scanLink(chnMap[i], chA, chB, xpmMeasFn, &xpm->_dsLinkStatus);
        if (idx > 16)  printf("No error-free zone found\n");
        else           printf("An error-free zone found centered at EQ index = %d\n", idx);

        links &= ~(1<<i);
      }
    }
  } else

  if (dump)
  {
    unsigned unit  = -1;
    unsigned links = linkEnables;
    for (unsigned i = 0; links; ++i)
    {
      if (links & (1<<i))
      {
        if (unit != i)
        {
          hsr[hsrMap[i]].dump(0);
          unit = i;
        }
        hsr[hsrMap[i]]._chB[chnMap[i]].dump();
        hsr[hsrMap[i]]._chA[chnMap[i]].dump();
        links &= ~(1<<i);
      }
    }
  }

  sleep(1);                             // Wait for cpsw threads to exit
}
