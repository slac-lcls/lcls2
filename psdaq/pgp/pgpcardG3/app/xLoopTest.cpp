
#include <sys/types.h>
#include <sys/mman.h>

#include <linux/types.h>
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
#include <stdint.h>

#include "../include/PgpCardMod.h"
#include "../include/PgpCardReg.h"

#define PAGE_SIZE 4096

using namespace std;

void showUsage(const char* p) {
  printf("Usage: %s [options]\n", p);
  printf("Options:\n"
         "\t-P <dev>   Use pgpcard <dev> (integer)\n"
         "\t-T <code>  Trigger on code/rate\n"
         "\t-L <lanes> Bit mask of enabled lanes\n"
         "\t-s <words> Tx size in 32b words\n"
         "\t-n <count> Number of transmissions in ring\n"
         "\t-D <delay> Trigger delay (0=8ns, 1=16ns, 2=32ns, 3=64ns, ..\n"
         "\t-F <count> TxFIFO low watermark\n"
         "\t-S         Simulate data\n");
}

int main (int argc, char **argv) {
  int           fd;
  int           ret;
  unsigned      idev=0;
  unsigned      lanes=1;
  unsigned      opCode=0;
  unsigned      size  =512;
  unsigned      ntx   =1;
  bool          lsim  =false;
  int           fifolo=-1;
  int           txReqDelay=-1;
  char dev[64];

  int c;

  while((c=getopt(argc,argv,"P:T:L:n:s:F:SD:")) != EOF) {
    switch(c) {
    case 'P': idev   = strtoul(optarg,NULL,0); break;
    case 'T': opCode = strtoul(optarg,NULL,0); break;
    case 'L': lanes  = strtoul(optarg,NULL,0); break;
    case 'F': fifolo = strtoul(optarg,NULL,0); break;
    case 'n': ntx    = strtoul(optarg,NULL,0); break;
    case 's': size   = strtoul(optarg,NULL,0); break;
    case 'S': lsim   = true; break;
    case 'D': txReqDelay = strtoul(optarg,NULL,0); break;
    default:
      showUsage(argv[0]); return 0;
    }
  }

  sprintf(dev,"/dev/pgpcardG3_%u_0",idev);
  if ( (fd = open(dev, O_RDWR)) <= 0 ) {
    cout << "Error opening " << dev << endl;
    perror(dev);
    return(1);
  }


  void volatile *mapStart;
  PgpReg* pgpReg;

  // Map the PCIe device from Kernel to Userspace
  mapStart = (void volatile *)mmap(NULL, PAGE_SIZE, (PROT_READ|PROT_WRITE), (MAP_SHARED|MAP_LOCKED), fd, 0);   
  if(mapStart == MAP_FAILED){
    cout << "Error: mmap() = " << dec << mapStart << endl;
    close(fd);
    return(1);   
  }

  pgpReg = (PgpReg*)mapStart;

  //  First, clear the FIFO of pending Tx DMAs
  unsigned txControl = pgpReg->txControl & ~((lanes<<16)|lanes);
  pgpReg->txControl = txControl | (lanes<<16);

  //  Second, load the FIFO with the new Tx data
  if (opCode) {
    PgpCardTx     pgpCardTx;
    time_t        t;
    unsigned      count=0;
    unsigned      vc=0;
    unsigned*     data;
    unsigned      x;
    int           lfd[8];

    time(&t);
    srandom(t);

    data = (uint *)malloc(sizeof(uint)*size);

    for(unsigned lane=0; lane<8; lane++) {
      if ( (lanes & (1<<lane))==0 )
        lfd[lane] = -1;
      else {
        sprintf(dev,"/dev/pgpcardG3_%u_%u",idev,lane+1);
        if ( (lfd[lane] = open(dev, O_RDWR)) <= 0 ) {
          cout << "Error opening " << dev << endl;
          perror(dev);
          return(-1);
        }
      }
    }

    pgpCardTx.cmd = IOCTL_Tx_Loop_Clear;
    pgpCardTx.model = sizeof(&pgpCardTx);
    pgpCardTx.size = sizeof(PgpCardTx);

    for(unsigned lane=0; lane<8; lane++) {
      if ( (lanes & (1<<lane))==0 )
        continue;

      pgpCardTx.data = (__u32*)(1<<lane);

      printf("Clearing Tx for lane %d\n",lane);
      write(lfd[lane],&pgpCardTx,sizeof(PgpCardTx));
    }

    pgpCardTx.model   = (sizeof(data));
    pgpCardTx.cmd     = lsim ? IOCTL_LoopSim_Write : IOCTL_Looped_Write;
    pgpCardTx.pgpVc   = vc;
    pgpCardTx.size    = size;
    pgpCardTx.data    = (__u32*)data;

    while (count++ < ntx) {
      // DMA Write
      cout << "Sending:";
      //      cout << " Lane=" << dec << lane;
      cout << ", Vc=" << dec << vc << endl;
      if (!lsim) {
        data[0] = count-1;
        for (x=1; x<size; x++) {
          data[x] = random();
          if (x<40) {
            cout << " 0x" << setw(8) << setfill('0') << hex << data[x];
            if ( ((x+1)%10) == 0 ) cout << endl << "   ";
          }
        }
        cout << endl;
      }
      for(unsigned lane=0; lane<8; lane++) {
        if ( (lanes & (1<<lane))==0 )
          continue;

        pgpCardTx.pgpLane = lane;
        ret = write(lfd[lane],&pgpCardTx,sizeof(PgpCardTx));
        cout << "Lane " << dec << lane << ": " << dec << ret << endl;
      }
    }
    free(data);
  }

  txControl |= (fifolo&0xf)<<28;
  if (txReqDelay>=0) {
    txControl &= ~(0xf<<24);
    txControl |= (txReqDelay&0xf)<<24;
  }

  //  Finally, enable the FIFO
  if (opCode) 
    pgpReg->txControl = txControl | lanes;
  else
    pgpReg->txControl = txControl;

  printf("TxControl: %08x\n", pgpReg->txControl);

  pgpReg->txOpCode = opCode | (lanes<<8);
  printf("TxOpCode: %08x\n", pgpReg->txOpCode);

  close(fd);
  return 0;
}
