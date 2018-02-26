
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
#include <pthread.h>
#include <linux/types.h>

#include "../include/DmaDriver.h"
#include "PgpDaq.hh"

using namespace std;

static void usage(const char* p)
{
  printf("Usage: %s <options>\n",p);
  printf("Options:\n");
  printf("\t-d <device>  [e.g. /dev/pgpdaq0]\n");
  printf("\t-c <client>  \n");
  printf("\t-L <lanes>   [mask of lanes]\n");
}

static uint32_t events = 0;
static uint64_t bytes  = 0;
static uint32_t misses = 0;

static void printrate(const char* name,
                      const char* units,
                      double      rate)
{
  static const char scchar[] = { ' ', 'k', 'M' };
  unsigned rsc = 0;

  if (rate > 1.e6) {
    rsc     = 2;
    rate   *= 1.e-6;
  }
  else if (rate > 1.e3) {
    rsc     = 1;
    rate   *= 1.e-3;
  }

  printf("%s %7.2f %c%s", name, rate, scchar[rsc], units);
}

static void* countThread(void* p)
{
  uint32_t pevents = events;
  uint32_t pmisses = misses;
  uint64_t pbytes  = bytes;

  timespec ptv;
  clock_gettime(CLOCK_REALTIME, &ptv);

  while(1) {
    sleep(1);
    timespec tv;
    clock_gettime(CLOCK_REALTIME, &tv);
    double dt = double(tv.tv_sec - ptv.tv_sec) + 1.e-9*(double(tv.tv_nsec)-double(ptv.tv_nsec));
    double revents = double(events-pevents)/dt;
    double rmisses = double(misses-pmisses)/dt;
    double rbytes  = double(bytes -pbytes )/dt;
    printrate("\n ", "Hz", revents);
    printrate("\t ", "B/s", rbytes);
    printrate("\t misses ", "Hz", rmisses);
    pevents = events;
    pmisses = misses;
    pbytes  = bytes;
    ptv     = tv;
  }

  return 0;
}

int main (int argc, char **argv) {

  int          fd;
  const char*  dev = "/dev/pgpdaq0";
  unsigned     client = 0;
  unsigned     lanes  = 1;
  bool         lverbose = false;
  int c;

  while((c=getopt(argc,argv,"d:c:L:")) != EOF) {
    switch(c) {
    case 'c': client = strtoul(optarg,NULL,0); break;
    case 'd': dev    = optarg; break;
    case 'L': lanes  = strtoul(optarg,NULL,0); break;
    default: usage(argv[0]); return 0;
    }
  }

  char cdev[64];
  sprintf(cdev,"%s_%u",dev,client);
  if ( (fd = open(cdev, O_RDWR)) <= 0 ) {
    cout << "Error opening " << cdev << endl;
    return(1);
  }

  //
  //  Launch the statistics thread
  //
  pthread_attr_t tattr;
  pthread_attr_init(&tattr);
  pthread_t thr;
  if (pthread_create(&thr, &tattr, &countThread, 0)) {
    perror("Error creating stat thread");
    return -1;
  }

  //
  //  Map the lanes to this reader
  //
  {
    PgpDaq::PgpCard* p = (PgpDaq::PgpCard*)mmap(NULL, sizeof(PgpDaq::PgpCard), (PROT_READ|PROT_WRITE), (MAP_SHARED|MAP_LOCKED), fd, 0);   
    uint32_t MAX_LANES = p->nlanes();
    for(unsigned i=0; i<MAX_LANES; i++)
      if (lanes & (1<<i))
        p->dmaLane[i].client = client;
  }

  //
  //  Launch the read loop
  //
  struct DmaReadData rd;
  rd.data  = reinterpret_cast<uintptr_t>(new char[0x200000]);
  unsigned index = 0;
  uint32_t qlast = 0;
  while(1) {
    //    usleep(1000);
    rd.index = 0;
    ssize_t sz = read(fd, &rd, sizeof(rd));
    if (sz < 0) {
      perror("Reading buffer");
      return -1;
    }
    //    printf("Read buffer of size 0x%x\n", rd.size);
    if (!rd.size) {
      misses++;
      continue;
    }

    if (lverbose) {
      const uint32_t* q = reinterpret_cast<const uint32_t*>(rd.data);
      printf("%08x [%d] [%x], idx %03x, dst %02x, flags %02x, err %01x, size %06x\n", 
             q[0], q[0]-qlast, index, rd.index, rd.dest, rd.flags, rd.error, rd.size);
      qlast = q[0];
    }

    events++;
    bytes += rd.size;
    index++;
  }

  pthread_join(thr,NULL);

  close(fd);
  return 0;
}
