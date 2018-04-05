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


unsigned measFn(void* arg)
{
  Cphw::Reg* rxErrs = (Cphw::Reg*)arg;

  unsigned e0 = *rxErrs;
  //usleep(90000);           // In 90 mS, the 186 MHz clock will count ~16.7M times
  //  usleep(125000);     // Let rxErrs roll over, but finish everything in ~2 sec.
  usleep(1250000);     // Let rxErrs roll over, but finish everything in ~2 sec.
  unsigned e1 = *rxErrs;
  return (((e1 & 0x00ffffff) - (e0 & 0x00ffffff)) + 0x1000000) & 0x00ffffff;
}


void usage(const char* p)
{
  printf("Usage: %s [options]\n",p);
  printf("Options: -a <IP addr (dotted notation)> : Use network <IP>\n");
  printf("         -p <port>                      : Use network <port>\n");
  printf("         -L <ds link mask>              : Enable selected link(s)\n");
  printf("         -M <mode>                      : Mode select for selected link(s)\n");
  printf("         -D <DEM value>                 : DEM value for selected link(s)\n");
  printf("         -V <VOD value>                 : VOD value for selected link(s)\n");
  printf("         -E <EQ value>                  : EQ value for selected link(s)\n");
  printf("         -B                             : Apply -E, -D and/or -V to B channel(s)\n");
  printf("         -A                             : Apply -E, -D and/or -V to A channel(s)\n");
  printf("         -P <value>                     : Set sigdet preset for selected link(s)\n");
  printf("         -r                             : Reset HSR registers of selected link(s)\n");
  printf("         -S                             : Reset SMBus Master of selected link(s)\n");
  printf("         -s                             : Scan EQ values of selected link(s) (overrides -E)\n");
  printf("         -e                             : Leave smbus enabled\n");
  printf("         -d                             : Dump registers of selected link(s)\n");
  printf("         -0                             : Power down links\n");
  printf("         -f <fname>                     : Save full configuration to file\n");
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
  int  sdp  = -1;
  int  pdn  = -1;
  int  mde  = -1;
  bool rstRegs  = false;
  bool rstSMBus = false;
  //  bool scan = false;
  bool dump = false;
  bool enable = false;
  const char* ofile=0;

  while ( (c=getopt( argc, argv, "a:f:p:L:D:V:E:M:P:BAerSsd01h")) != EOF ) {
    switch(c) {
    case 'a':
      ip = optarg; break;
      break;
    case 'p':
      port = strtoul(optarg,NULL,0);
      break;
    case 'f':
      ofile = optarg;
      break;
    case 'e':
      enable = true;
      break;
    case 'L':
      linkEnables = strtoul(optarg,NULL,0) & ((1 << NLinks) - 1);
      break;
    case 'P':
      sdp = strtoul(optarg,NULL,0);
      break;
    case 'D':
      dem = strtoul(optarg,NULL,0);
      break;
    case 'M':
      mde = strtoul(optarg,NULL,0);
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
    case 'r':
      rstRegs = true;
      break;
    case 'S':
      rstSMBus = true;
      break;
    case 's':
      //      scan = true;
      break;
    case 'd':
      dump = true;
      break;
    case '0':
      pdn = 1;
      break;
    case '1':
      pdn = 0;
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

  Pds::Cphw::AmcTiming* t = new(0) Pds::Cphw::AmcTiming;
  printf("buildStamp %s\n",t->version.buildStamp().c_str());

  HsRepeater*  hsr = new((void*)0x09000000) Pds::Cphw::HsRepeater;

  if (sdp >= 0)
  {
    unsigned links = linkEnables;
    for (unsigned i = 0; links; ++i)
    {
      if (links & (1<<i))
      {
        hsr[hsrMap[i]].smbusEnable(true);
        if (chB)
          hsr[hsrMap[i]]._chB[chnMap[i]].sigDetPreset(sdp);
        if (chA)
          hsr[hsrMap[i]]._chA[chnMap[i]].sigDetPreset(sdp);
        hsr[hsrMap[i]].smbusEnable(false);
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

  if (mde >= 0)
  {
    unsigned links = linkEnables;
    for (unsigned i = 0; links; ++i)
    {
      if (links & (1<<i))
      {
        hsr[hsrMap[i]].smbusEnable(true);
        if (chB)
          hsr[hsrMap[i]]._chB[chnMap[i]].mode_sel(mde);
        if (chA)
          hsr[hsrMap[i]]._chA[chnMap[i]].mode_sel(mde);
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

  if (pdn>=0)
    {
    unsigned links = linkEnables;
    for (unsigned i = 0; links; ++i)
      {
        if (links & (1<<i)) {
          hsr[hsrMap[i]].smbusEnable(true);
          hsr[hsrMap[i]].pwdnOverride(true);
          for(unsigned j=0; j<8; j++)
            hsr[hsrMap[i]].pwdnChan(j,pdn);
          hsr[hsrMap[i]].smbusEnable(false);
        }
        links &= ~(1<<i);
      }
    }

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

  if (enable) 
    {
      unsigned links = linkEnables;
      for (unsigned i = 0; links; ++i) {
        if (links & (1<<i))
          hsr[hsrMap[i]].smbusEnable(true);
        links &= ~(1<<i);
      }
    }

  if (ofile) {
    FILE* f = fopen(ofile,"w");
    for(unsigned i=0; i<6; i++)
      hsr[i].save(f);
    printf("Configuration saved to %s\n",ofile);
  }

  sleep(1);                             // Wait for cpsw threads to exit
}
