
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
#include <signal.h>
#include <pthread.h>
#include <linux/types.h>

#include "../include/DmaDriver.h"
#include "PgpDaq.hh"

using namespace std;

enum { SameSize, LaneCheck, MonoPktIdx, ContPktIdx,
       MonoPid, FixedPid,
       MonoTS,
       ContEvtCnt, 
       AnyEvent,
};

static const char* checkDesc[] = { "same size",
                                   "lane check",
                                   "monotonic packet index",
                                   "continuous packet index",
                                   "monotonic pulse ID",
                                   "fixed pulse ID increment",
                                   "monotonic timestamp",
                                   "continous event count",
                                   "any",
                                   NULL };

static void usage(const char* p)
{
  printf("Usage: %s <options>\n",p);
  printf("Options:\n");
  printf("\t-d <device>  [e.g. /dev/pgpdaq0]\n");
  printf("\t-c <client>  \n");
  printf("\t-L <lanes>   [mask of lanes]\n");
  printf("\t-S <nanosec> [sleep each event]\n");
  printf("\t-P <nprint>  [events to print]\n");
  printf("\t-E           [expect event header]\n");
  printf("\t-V <mask>    [print invalidate]\n");
  for(unsigned i=0; checkDesc[i]!=NULL; i++)
    printf("\t\t b%u : %s\n", i, checkDesc[i]);
}

static uint32_t events = 0;
static uint64_t bytes  = 0;
static uint32_t misses = 0;
static uint32_t drops  = 0;
static uint32_t excepts= 0;
static uint32_t xmask  = 0;
static uint32_t idxhst[0x40];

static void dumphist(const uint32_t* p,
                     const unsigned  n)
{
  for(int j=7; j>=0; j--) {
    for(unsigned i=0; i<n; i++) {
      unsigned v = p[i];
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
    printf("%x...............", i);
  printf("\n");

  printf("Events: %u\n"  , events);
  printf("Bytes : %llu\n", bytes);
  printf("Misses: %u\n"  , misses);
  printf("Drops : %u\n"  , drops);
  printf("Excpts: %u\n"  , excepts);
}

static void sigHandler(int signal)
{
  printf("-- Buffer index usage --\n");
  dumphist(idxhst, 0x40);
  ::exit(signal);
}

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
  uint32_t pdrops  = drops;
  uint64_t pbytes  = bytes;
  uint32_t pexcepts= excepts;

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
    printf("\t drops %u", (drops-pdrops));
    printf("\t excepts %u", (excepts-pexcepts));
    printf("\t emask %x", xmask);
    pevents = events;
    pmisses = misses;
    pdrops  = drops;
    pbytes  = bytes;
    pexcepts= excepts;
    ptv     = tv;
    xmask   = 0;
  }

  return 0;
}

static void nanospin(unsigned ns)
{
  timespec tv;
  clock_gettime(CLOCK_REALTIME,&tv);
  tv.tv_nsec += ns;
  if (tv.tv_nsec >= 1000000000) {
    tv.tv_sec++;
    tv.tv_nsec -= 1000000000;
  }
  while(1) {
    timespec now;
    clock_gettime(CLOCK_REALTIME,&now);
    if (now.tv_sec > tv.tv_sec)
      break;
    if (now.tv_sec < tv.tv_sec)
      continue;
    if (now.tv_nsec > tv.tv_nsec)
      break;
  }
}

int main (int argc, char **argv) {

  int          fd;
  const char*  dev = "/dev/pgpdaq0";
  unsigned     client = 0;
  unsigned     lanes  = 1;
  unsigned     nprint = 0;
  unsigned     vmask  = 0;
  unsigned     nsSleep = 0;
  bool         lEventHdr = false;
  int c;

  while((c=getopt(argc,argv,"d:c:L:P:S:V:E")) != EOF) {
    switch(c) {
    case 'c': client  = strtoul(optarg,NULL,0); break;
    case 'd': dev     = optarg; break;
    case 'E': lEventHdr = true; break;
    case 'L': lanes   = strtoul(optarg,NULL,0); break;
    case 'P': nprint  = strtoul(optarg,NULL,0); break;
    case 'S': nsSleep = strtoul(optarg,NULL,0); break;
    case 'V': vmask   = strtoul(optarg,NULL,0); break;
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
      if (lanes & (1<<i)) {
        p->dmaLane[i].client = client;
        p->pgpLane[i].axil.txControl = 1;  // disable flow control
      }
  }

  memset(idxhst, 0, 0x40*sizeof(uint32_t));

  ::signal( SIGINT , sigHandler );
  ::signal( SIGABRT, sigHandler );
  ::signal( SIGKILL, sigHandler );

  //
  //  Launch the read loop
  //
  struct DmaReadData rd;
  rd.data  = reinterpret_cast<uintptr_t>(new char[0x200000]);
  unsigned index = 0;
  unsigned slast[4];
  memset(slast, 0, sizeof(slast));
  unsigned clast[4];
  memset(clast, 0, sizeof(clast));
  uint64_t pidLast[4];
  memset(pidLast, 0, sizeof(pidLast));
  uint64_t dpidLast[4];
  memset(dpidLast, 0, sizeof(dpidLast));
  uint64_t tsLast[4];
  memset(tsLast, 0, sizeof(tsLast));
  unsigned ecLast[4];
  memset(ecLast, 0, sizeof(ecLast));

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

    unsigned emask=(1<<AnyEvent);
    unsigned lane = (rd.dest>>5)&7;
    //
    //  Validate size
    //
    if (rd.size != slast[lane]) {
      emask |= (1<<SameSize);
      slast[lane] = rd.size;
    }

    //
    //  Validate received lane  
    //
    if (((1<<lane) & lanes)==0) {
      emask |= (1<<LaneCheck);
    }

    const uint32_t* q = reinterpret_cast<const uint32_t*>(rd.data);
    //
    //  Validate event header
    //
    if (lEventHdr) {
      uint64_t pid = *reinterpret_cast<const uint64_t*>(&q[0]);
      if ((pid>>56)!=0) {
        //  Datagram is a non-L1 transition
        printf("\nNon-L1 [%x]", (pid>>56));
        for(unsigned i=0; i<8; i++)
          printf("%c%08x", (i==0) ? ' ':':', q[i]);
        continue;
      }

      //
      //  Pulse ID is monotonic
      //
      uint64_t dpid = pid-pidLast[lane];
      if (pid < pidLast[lane])
        emask |= (1<<MonoPid);
      pidLast[lane] = pid;

      //
      //  Pulse ID increments at fixed interval
      //
      if (dpid != dpidLast[lane]) {
        emask |= (1<<FixedPid);
        dpidLast[lane] = dpid;
      }

      //
      //  Timestamp is monotonic
      //
      uint64_t ts = *reinterpret_cast<const uint64_t*>(&q[2]);
      if (ts < tsLast[lane])
        emask |= (1<<MonoTS);
      tsLast[lane] = ts;

      //
      //  Event counter is continuous
      //
      uint32_t ec = q[4]&0xffffff;
      uint32_t dec = ec - ecLast[lane];
      if (dec != 1)
        emask |= (1<<ContEvtCnt);
      ecLast[lane] = ec;

      //
      //  partitions & trigger lines
      //

      if (lEventHdr && (nprint || (emask&vmask))) {
        printf("\n dPID[%llx] dEC[%x]", dpid, dec);
        for(unsigned i=0; i<8; i++)
          printf("%c%08x", (i==0) ? ' ':':', reinterpret_cast<const uint32_t*>(rd.data)[i]);
      }

      q += 8;
    }

    //
    //  Validate packet counter
    //
    if (q[0] < clast[lane]) {
      emask |= (1<<MonoPktIdx);
    }
    else if (q[0] != clast[lane]+1) {
      emask |= (1<<ContPktIdx);
      drops += q[0]-clast[lane]-1;
    }

    if (emask & 0xff) {
      excepts++;
    }

    if (nprint || (emask&vmask)) {
      printf("\n%08x:%08x [%d] [%x], idx %03x, dst %02x, flags %02x, err %01x, size %06x, except %02x", 
             q[0], q[1], q[0]-clast[lane], index, rd.index, rd.dest, rd.flags, rd.error, rd.size, emask);
      if (nprint) {
        printf("\n");
        for(unsigned i=0; i<(rd.size>>2); i++)
          printf("%08x%c",q[i],(i%8)==7 ? '\n':' ');
        nprint--;
      }
    }

    clast[lane] = q[0];
    idxhst[rd.index]++;

    events++;
    bytes += rd.size;
    index++;
    xmask |= emask;

    if (nsSleep)
      nanospin(nsSleep);
  }

  pthread_join(thr,NULL);

  close(fd);
  return 0;
}
