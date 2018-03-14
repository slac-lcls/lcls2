
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

#include "PgpDaq.hh"

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

static void dump_by2(const uint32_t* p,
                     const unsigned  n)
{
      for(int j=7; j>=0; j--) {
        for(unsigned i=0; i<n/2; i++) {
          unsigned v = p[2*i+0] + p[2*i+1];
          v >>= 4*j;
          unsigned d = v&0xf;
          if (v)
            printf("%01x",d);
          else
            printf(" ");
        }
        printf("\n");
      }
      for(unsigned i=0; i<n/16; i++)
        printf("%x.......", i);
      printf("\n");
}

static void dump_by4(const uint32_t* p,
                     const unsigned  n)
{
      for(int j=7; j>=0; j--) {
        for(unsigned i=0; i<n/4; i++) {
          unsigned v = 0;
          v += p[4*i+0];
          v += p[4*i+1];
          v += p[4*i+2];
          v += p[4*i+3];
          v >>= 4*j;
          unsigned d = v&0xf;
          if (v)
            printf("%01x",d);
          else
            printf(" ");
        }
        printf("\n");
      }
      for(unsigned i=0; i<n/16; i++)
        printf("%x...", i);
      printf("\n");
}

class AxisHistogramT {
public:
  AxisHistogramT(const std::string pvname) : _name(pvname)
  {
  }
public:
  const uint32_t* update(const uint32_t* p) {
    if ((p[0]>>8) != 0xbdbdbd) {
      printf("Header not found (%x)\n",p[0]);
      return 0;
    }
    unsigned sz = 1<<(p[0]&0x1f);
    if (1) {
      for(unsigned i=0; i<sz; i++) {
        _diff[i] = p[i+1]-_data[i];
        _data[i] = p[i+1];
      }
      // printf("%s: sum\n",_name.c_str());
      // dump_by2(_data, 256);
      printf("%s: diff\n",_name.c_str());
      dump_by2(_diff, 256);
    }
    return p + sz + 1;
  }
private:
  std::string _name;
  uint32_t _data[256];
  uint32_t _diff[256];
};

//
// Address Map, offset from base
//
   /* constant VERSION_ADDR_C : slv(31 downto 0) := x"00000000"; */
   /* constant PHY_ADDR_C     : slv(31 downto 0) := x"00010000"; */
   /* constant BPI_ADDR_C     : slv(31 downto 0) := x"00030000"; */
   /* constant SPI0_ADDR_C    : slv(31 downto 0) := x"00040000"; */
   /* constant SPI1_ADDR_C    : slv(31 downto 0) := x"00050000"; */
   /* constant APP_ADDR_C     : slv(31 downto 0) := x"00800000"; */
//  MigToPcieWrapper : x"00800000"
//  HardwareSemi     : x"00C00000"
//

static void usage(const char* p)
{
  printf("Usage: %p <options>\n",p);
  printf("Options:\n");
  printf("\t-d <device>        defaults to /dev/pgpdaq0\n");
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
  const char*  base = "DAQ:PGP:";
  unsigned     sample_period = 0;
  unsigned     readout_period = 0;
  bool         lEnable = false;
  bool         lDisable = false;
  int c;

  while((c=getopt(argc,argv,"d:mML:P:s:r:")) != EOF) {
    switch(c) {
    case 'd': dev = optarg; break;
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
      std::stringstream o;                                      \
      o << base << "LANE" << i << parm;                         \
      hist.push_back(new AxisHistogramT(o.str())); }

    const unsigned NLINKS = p->nlanes();
    const unsigned NAPPS  = p->nclients();

    std::list<AxisHistogramT*> hist;
    for(unsigned i=0; i<NLINKS; i++)
      LANE_HIST(i,"FREEBLKS");
    for(unsigned i=0; i<NAPPS; i++)
      LANE_HIST(i,"FREEDESCS");

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
      for(std::list<AxisHistogramT*>::iterator it = hist.begin(); it!=hist.end() && q!=0; it++)
        q = (*it)->update(q);
    }
  }

  close(fd);
  return 0;
}
