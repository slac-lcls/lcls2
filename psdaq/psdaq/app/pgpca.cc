
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <termios.h>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <linux/types.h>

#include <list>
#include <strstream>

#include "psdaq/pgp/pgpGen4Daq/app/PgpDaq.hh"
#include "psdaq/epicstools/PVWriter.hh"

using Pds_Epics::PVWriter;
using namespace std;

// RX Structure
// Data = 0 for read index
struct DmaReadData {
   uint64_t   data;
   uint32_t   dest;
   uint32_t   flags;
   uint32_t   index;
   uint32_t   error;
   uint32_t   size;
   uint32_t   is32;
};

class AxisHistogram {
public:
  AxisHistogram(const char* pvname) :
    _pv   (new PVWriter(pvname))
  {
  }
public:
  const uint32_t* update(const uint32_t* p) {
    if ((p[0]>>8) != 0xbdbdbd) {
      printf("Header not found (%x)\n",p[0]);
      return 0;
    }
    unsigned sz = 1<<(p[0]&0x1f);
    if (_pv->connected()) {
      unsigned csz = sz*sizeof(uint32_t);
      if (_pv->data_size() != csz) {
        printf("AxisHistogram: size disagrees [0x%zx:0x%x]\n",
               _pv->data_size(), csz);
      }
      else {
        memcpy( _pv->data(), &p[1], csz);
        _pv->put();
      }
    }
    return p + sz + 1;
  }
private:
  PVWriter* _pv;
};

static void usage(const char* p)
{
  printf("Usage: %p <options>\n",p);
  printf("Options:\n");
  printf("\t-M                 enable monitoring\n");
  printf("\t-m                 disable monitoring\n");
  printf("\t-L                 lanes to map to app 0\n");
  printf("\t-P <threshold>     set pause threshold\n");
  printf("\t-s <sample scale>  set sampling period\n");
  printf("\t-r <readout scale> set readout period\n");
}

int main (int argc, char **argv) {

  int          fd;
  const char*  dev = "/dev/pgpdaq0";
  const char*  base = "DAQ:LAB2:PGPCARD:";
  unsigned     sample_period = 0;
  unsigned     readout_period = 0;
  bool         lEnable = false;
  bool         lDisable = false;
  int c;

  while((c=getopt(argc,argv,"mML:P:s:r:")) != EOF) {
    switch(c) {
    case 'm': lDisable = true; break;
    case 'M': lEnable = true; break;
    case 's': sample_period = strtoul(optarg, NULL, 0); break;
    case 'r': readout_period = strtoul(optarg, NULL, 0); break;
    default: usage(argv[0]); return 0;
    }
  }

  if ( (fd = open(dev, O_RDWR)) <= 0 ) {
    cout << "Error opening " << dev << endl;
    return(1);
  }

  PgpDaq::PgpCard* p = (PgpDaq::PgpCard*)mmap(NULL, 0x01000000, (PROT_READ|PROT_WRITE), (MAP_SHARED|MAP_LOCKED), fd, 0);   

  if (lDisable) {
    p->monEnable = 0;
  }
  else if (lEnable) {
    if (sample_period)
      p->monSampleInterval  = sample_period;
    if (readout_period)
      p->monReadoutInterval = readout_period;

    p->monEnable = 1;

#define LANE_HIST(i,parm) {                                     \
      ostrstream o;                                             \
      o << base << parm << i;                                   \
      hist.push_back(new AxisHistogram(o.str())); }

    const unsigned NLINKS = 4;
    const unsigned NAPPS  = 4;

    std::list<AxisHistogram*> hist;
    for(unsigned i=0; i<NLINKS; i++)
      LANE_HIST(i,"BLKSFREE");
    for(unsigned i=0; i<NAPPS; i++)
      LANE_HIST(i,"DESCFREE");
    ca_pend_io(0);

    struct DmaReadData rd;
    rd.data  = reinterpret_cast<uintptr_t>(new char[0x200000]);
    rd.index = 1<<31;
    while(1) {
      sleep(1);
      ssize_t sz = read(fd, &rd, sizeof(rd));
      if (sz < 0) {
        perror("Reading buffer");
        return -1;
      }
      const uint32_t* q = reinterpret_cast<const uint32_t*>(rd.data);
      for(std::list<AxisHistogram*>::iterator it = hist.begin(); it!=hist.end() && q!=0; it++)
        q = (*it)->update(q);
      ca_flush_io();
    }
  }

  close(fd);
  return 0;
}
