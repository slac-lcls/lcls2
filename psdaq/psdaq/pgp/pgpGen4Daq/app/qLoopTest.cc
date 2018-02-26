
#include <sys/types.h>
#include <sys/ioctl.h>
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
#include <sys/mman.h>

#include "PgpDaq.hh"

using namespace std;

void showUsage(const char* p) {
  printf("Usage: %s [options]\n", p);
  printf("Options:\n"
         "\t-d <dev>   Use device <dev> (integer)\n"
         "\t-T <code>  Trigger at fixed interval\n"
         "\t   0x00-0x0f = 156   - 10    MHz  [  1 - 16 clks]\n"
         "\t   0x10-0x1f =  39   -  2.5  MHz  [  4  - 64 clks]\n"
         "\t   0x20-0x2f =  10   -  0.6  MHz  [ 16  - 256 clks]\n"
         "\t   0x31-0x3f =   2.5 -  0.15 MHz  [ 64  - 1024 clks]\n"
         "\t   0x41-0x4f = 600   -  38   kHz  [256  - 4k clks]\n"
         "\t   0x51-0x5f = 152   -  9.5  kHz  [  1k - 16k clks]\n"
         "\t   0x61-0x6f =  38   -  2.4  kHz  [  4k - 64k clks]\n"
         "\t   0x71-0x7f =   9.5 -  0.6  kHz  [ 16k - 256k clks]\n"
         "\t-L <lanes> Bit mask of enabled lanes\n"
         "\t-s <words> Tx size in 32b words\n"
         "\t-D <delay> Trigger delay (0=8ns, 1=16ns, 2=32ns, 3=64ns, ..\n"
         "\t-F <count> TxFIFO low watermark\n"
         "\t-l <0/1>   set phy loopback\n");
}

int main (int argc, char **argv) {

  int           fd;
  unsigned      lanes=1;
  unsigned      opCode=0x80;
  unsigned      size  =512;
  int           fifolo=4;
  int           txReqDelay=0;
  int           loopb =-1;
  const char*   dev = "/dev/pgpdaq0";
  int c;

  while((c=getopt(argc,argv,"d:T:L:l:n:s:F:SD:")) != EOF) {
    switch(c) {
    case 'd': dev    = optarg; break;
    case 'T': opCode = strtoul(optarg,NULL,0); break;
    case 'L': lanes  = strtoul(optarg,NULL,0); break;
    case 'l': loopb  = strtoul(optarg,NULL,0); break;
    case 'F': fifolo = strtoul(optarg,NULL,0); break;
    case 's': size   = strtoul(optarg,NULL,0); break;
    case 'D': txReqDelay = strtoul(optarg,NULL,0); break;
    default:
      showUsage(argv[0]); return 0;
    }
  }

  if ( (fd = open(dev, O_RDWR)) <= 0 ) {
    cout << "Error opening " << dev << endl;
    return(1);
  }

  PgpDaq::PgpCard* p = (PgpDaq::PgpCard*)mmap(NULL, 0x01000000, (PROT_READ|PROT_WRITE), (MAP_SHARED|MAP_LOCKED), fd, 0);   

  if (loopb >= 0)
    for(unsigned i=0; i<4; i++)
      p->pgpLane[i].loopback = (loopb & (1<<i)) ? (2<<16) : 0;

  uint32_t control = p->sim.control;
  printf("AppTxSim control = %08x\n", control);

  control = 
    ((txReqDelay&0x0f)<<24) |
    ((fifolo&0x0f)<<28);

  p->sim.control = control; // clear fixed interval count
  p->sim.size    = size;    // set Tx size

  control |= 
    ((lanes&0xff)<<0) |
    ((opCode > 0x7f ? 0 : ((opCode<<1)|1))<<8);

  p->sim.control = control;

  printf("AppTxSim control = %08x\n", control);

  close(fd);
}
