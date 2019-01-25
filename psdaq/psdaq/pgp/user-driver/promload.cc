/**
 *-----------------------------------------------------------------------------
 * Title      : PGP Firmware Update Utility
 * ----------------------------------------------------------------------------
 * File       : pgpPromLoad.cpp
 * Author     : Ryan Herbst, rherbst@slac.stanford.edu
 * Created    : 2016-08-08
 * Last update: 2016-08-08
 * ----------------------------------------------------------------------------
 * Description:
 * Utility to program the PGP card with new firmware.
 * ----------------------------------------------------------------------------
 * This file is part of the aes_stream_drivers package. It is subject to 
 * the license terms in the LICENSE.txt file found in the top-level directory 
 * of this distribution and at: 
    * https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
 * No part of the aes_stream_drivers package, including this file, may be 
 * copied, modified, propagated, or distributed except according to the terms 
 * contained in the LICENSE.txt file.
 * ----------------------------------------------------------------------------
**/

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <linux/types.h>

#include <fcntl.h>
#include <sstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <argp.h>

#include "pgpdriver.h"
#include "aximicronn25q.hh"

using namespace std;

static void usage(const char* p)
{
  printf("Usage: %s [options]\n",p);
  printf("Options: -d <deviceId>\n"
         "         -b <busId>\n"
         "         -f <mcs file>\n"
         "         -r [read only]\n");
  printf("For multiple PROMs, use multiple -f <mcs file> arguments\n");
}

int main (int argc, char **argv) 
{
   AxisG2Device* dev = 0;
   bool lReadOnly = false;
   std::vector<std::string> mcs_files;
   int c;

  while((c=getopt(argc,argv,"d:b:f:r")) != EOF) {
    switch(c) {
    case 'd': dev = new AxisG2Device(strtoul(optarg,NULL,0)); break;
    case 'b': dev = new AxisG2Device(optarg); break;
    case 'f': mcs_files.push_back(std::string(optarg)); break;
    case 'r': lReadOnly = true; break;
    default: usage(argv[0]); return 0;
    }
  }

  if (!dev) {
    usage(argv[0]);
    return 0;
  }

  if (mcs_files.size()>0) {
    AxiMicronN25Q prom(reinterpret_cast<char*>(dev->reg()+0x00040000),
                       mcs_files[0].c_str());
    if (!lReadOnly)
      prom.load();
    prom.verify();
  }
  if (mcs_files.size()>1) {
    AxiMicronN25Q prom(reinterpret_cast<char*>(dev->reg()+0x00050000),
                       mcs_files[1].c_str());
    if (!lReadOnly)
      prom.load();
    prom.verify();
  }
            
  return 0;
}
