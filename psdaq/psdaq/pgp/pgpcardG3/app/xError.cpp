
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
#include <semaphore.h>

#include "../include/PgpCardMod.h"
#include "../include/PgpCardReg.h"

#define PAGE_SIZE 4096

using namespace std;

int main (int argc, char **argv) {
  int           fd;
  int           x;
  const char*  dev = "/dev/pgpcardG3_0_0";

  if (argc>2 || 
      (argc==2 && argv[1][0]=='-')) {
    printf("Usage: %s [<device>]\n", argv[0]);
    return(0);
  }
  else if (argc==2)
    dev = argv[1];

  if ( (fd = open(dev, O_RDWR)) <= 0 ) {
    cout << "Error opening " << dev << endl;
    return(1);
  }

  void volatile *mapStart;

  // Map the PCIe device from Kernel to Userspace
  mapStart = (void volatile *)mmap(NULL, PAGE_SIZE, (PROT_READ|PROT_WRITE), (MAP_SHARED|MAP_LOCKED), fd, 0);   
  if(mapStart == MAP_FAILED){
    cout << "Error: mmap() = " << dec << mapStart << endl;
    close(fd);
    return(1);   
  }

  PgpReg* reg = (PgpReg*)mapStart;

  unsigned errSum[NUMBER_OF_LANES], errLst[NUMBER_OF_LANES];
  unsigned loopCnt=0;

  for (x=0; x < NUMBER_OF_LANES; x++) {
    unsigned tmp = (reg->pgpLaneStat[x] >> 28)&0xF;
    errLst[x] = tmp;
    errSum[x] = 0;
  }

  while(1) {

    usleep(1000);
    for (x=0; x < NUMBER_OF_LANES; x++) {
      unsigned tmp = (reg->pgpLaneStat[x] >> 28)&0xF;
      errSum[x] += (tmp - errLst[x])&0xF;
      errLst[x] = tmp;
    }

    if ((++loopCnt % 1000)==0) {
      for (x=0; x < NUMBER_OF_LANES; x++) {
        printf(" %8u", errSum[x]);
        errSum[x]=0;
      }
      printf("\n");
    }
  }

  return 0;
}
