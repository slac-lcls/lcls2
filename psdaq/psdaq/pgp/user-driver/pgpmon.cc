#include <chrono>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <pthread.h>
#include <list>
#include <sstream>
#include <signal.h>
#include "pgpdriver.h"

static AxisG2Device* pdev = 0;

static void sigHandler( int signal ) {
  psignal( signal, "Signal received by pgpmon application");
  if (pdev)
    pdev->exitmon();
  ::exit(signal);
}

static void dump_by2(const uint32_t* p,
                     const unsigned  n)
{
      for(int j=7; j>=0; j--) {
        for(unsigned i=0; i<n/2; i++) {
          uint32_t v = p[2*i+0] + p[2*i+1];
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
      bool lnodiff=true;
      uint32_t diff[256];
      for(unsigned i=0; i<sz; i++) {
        if (diff[i] = p[i+1]-_data[i])
          lnodiff=false;
        // if (_data[i] > p[i+1])
        //   printf("[%02x] (%04x - %04x : %04x)\n", i, p[i+1],_data[i],_diff[i]);
        _data[i] = p[i+1];
      }
      if (!lnodiff)
        memcpy(_diff,diff,256*sizeof(uint32_t));

      // printf("%s: sum\n",_name.c_str());
      // dump_by2(_data, 256);
      printf("%s: %s\n",_name.c_str(), lnodiff ? "nodiff":"diff");
      dump_by2(_diff, 256);
    }
    return p + sz + 1;
  }
private:
  std::string _name;
  uint32_t _data[256];
  uint32_t _diff[256];
};


int main(int argc, char* argv[])
{
    
    int c;
    int device_id;
    bool lverbose = false;
    while((c = getopt(argc, argv, "d:v")) != EOF) {
        switch(c) {
        case 'd':
          device_id = std::stoi(optarg, nullptr, 16);
          break;
        case 'v':
          lverbose = true;
          break;
        }
    }

    AxisG2Device dev(device_id);
    void* pmon = dev.initmon();
    const char*  base = "DAQ:PGP:";

    pdev = &dev;
    ::signal( SIGINT,  sigHandler );

#define LANE_HIST(i,parm) {                                     \
      std::stringstream o;                                      \
      o << base << "LANE" << i << parm;                         \
      hist.push_back(new AxisHistogramT(o.str())); }

    const unsigned NLINKS = ( get_reg32(dev.reg(), RESOURCES) >> 0 ) & 0xf;
    const unsigned NAPPS  = ( get_reg32(dev.reg(), RESOURCES) >> 4 ) & 0xf;

    std::list<AxisHistogramT*> hist;
    for(unsigned i=0; i<NLINKS; i++)
      LANE_HIST(i,"FREEBLKS");
    for(unsigned i=0; i<NAPPS; i++)
      LANE_HIST(i,"FREEDESCS");

    while (true) {    
        
      sleep(1);

      const uint32_t* q = reinterpret_cast<const uint32_t*>(pmon);
      for(std::list<AxisHistogramT*>::iterator it = hist.begin(); it!=hist.end() && q!=0; it++)
        q = (*it)->update(q);

    }                                

  return 0;
} 
