
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <arpa/inet.h>
#include <poll.h>
#include <signal.h>
#include <new>

#include "psdaq/hsd/Hsd3200.hh"
#include "psdaq/hsd/Globals.hh"
#include "psdaq/hsd/TprCore.hh"
#include "psdaq/hsd/QABase.hh"
#include "psdaq/mmhw/RingBuffer.hh"

using Pds::Mmhw::RingBuffer;

#include <string>

extern int optind;

using namespace Pds::HSD;

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -d <dev id>\n");
  printf("\t-C <initialize clock synthesizer>\n");
  printf("\t-D <add debug timer>\n");
  printf("\t-R <reset timing frame counters>\n");
  printf("\t-S <sync ADC>\n");
  printf("\t-U <sync clocktree>\n");
  printf("\t-X <reset gtx timing receiver>\n");
  printf("\t-Y <reset gtx timing transmitter>\n");
  printf("\t-Z <reset 625M PLL>\n");
  printf("\t-P <reverse gtx rx polarity>\n");
  printf("\t-0 <dump raw timing receive buffer>\n");
  printf("\t-1 <dump timing message buffer>\n");
  //  printf("Options: -a <IP addr (dotted notation)> : Use network <IP>\n");
}

void* debugThread(void*);

int main(int argc, char** argv) {

  extern char* optarg;
  char* endptr;

  const char* devname = "/dev/qadca";
  int c;
  bool lUsage = false;
  bool lSetupClkSynth = false;
  bool lReset = false;
  bool lResetRx = false;
  bool lResetTx = false;
  bool lPolarity = false;
  bool lRing0 = false;
  bool lRing1 = false;
  bool lClkSync = false;
  bool lAdcSync = false;
  bool lAdcSyncRst = false;

#if 0
  bool lSetPhase = false;
  unsigned delay_int=0, delay_frac=0;
#endif

  while ( (c=getopt( argc, argv, "CDRXYP01d:hSUZ")) != EOF ) {
    switch(c) {
    case 'C':
      lSetupClkSynth = true;
      break;
    case 'D':
      { pthread_attr_t tattr;
        pthread_attr_init(&tattr);
        pthread_t thr;
        if (pthread_create(&thr, &tattr, &debugThread, 0))
          perror("Error creating debug thread");
      }
      break;
    case 'P':
      lPolarity = true;
      break;
    case 'R':
      lReset = true;
      break;
    case 'X':
      lResetRx = true;
      break;
    case 'Y':
      lResetTx = true;
      break;
    case '0':
      lRing0 = true;
      break;
    case '1':
      lRing1 = true;
      break;
    case 'd':
      devname = optarg;
      break;
    case 'U':
      lClkSync = true;
      break;
    case 'S':
      lAdcSync = true;
      break;
    case 'Z':
      lAdcSyncRst = true;
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

  int fd = open(devname, O_RDWR);
  if (fd<0) {
    perror("Open device failed");
    return -1;
  }

  Hsd3200* p = Hsd3200::create(fd);

  p->dumpMap();

  p->board_status();

  printf("TPR [%p]\n", &(p->tpr()));
  p->tpr().dump();

  //  p->fmc_init();

  p->setup_timing(LCLSII);
  
  if (lPolarity) {
    p->tpr().rxPolarity(!p->tpr().rxPolarity());
  }

  if (lResetRx) {
    if (lSetupClkSynth)
      sleep(1);

    p->tpr().setLCLSII();
    p->tpr().resetRxPll();
    usleep(1000000);
    //    p->tpr().resetRx();
    p->tpr().resetBB();
  }

  if (lResetTx) {
    QABase& base = *reinterpret_cast<QABase*>((char*)p->reg()+0x80000);
    base.resetFbPLL();
    usleep(100000);
    base.resetFb();
  }

  if (lClkSync) {
    p->fmc_init();
    p->clocktree_sync();
  }

  if (lAdcSync) {
    p->sync();
  }

  if (lAdcSyncRst) {
    QABase& base = *reinterpret_cast<QABase*>((char*)p->reg()+0x80000);
    base.resetClock(true);
  }

  p->fmc_dump();

  if (lReset)
    p->tpr().resetCounts();

  printf("TPR [%p]\n", &(p->tpr()));
  p->tpr().dump();

  for(unsigned i=0; i<5; i++) {
    timespec tvb;
    clock_gettime(CLOCK_REALTIME,&tvb);
    unsigned vvb = p->tpr().TxRefClks;

    usleep(10000);

    timespec tve;
    clock_gettime(CLOCK_REALTIME,&tve);
    unsigned vve = p->tpr().TxRefClks;
    
    double dt = double(tve.tv_sec-tvb.tv_sec)+1.e-9*(double(tve.tv_nsec)-double(tvb.tv_nsec));
    printf("TxRefClk rate = %f MHz\n", 16.e-6*double(vve-vvb)/dt);
  }

  for(unsigned i=0; i<5; i++) {
    timespec tvb;
    clock_gettime(CLOCK_REALTIME,&tvb);
    unsigned vvb = p->tpr().RxRecClks;

    usleep(10000);

    timespec tve;
    clock_gettime(CLOCK_REALTIME,&tve);
    unsigned vve = p->tpr().RxRecClks;
    
    double dt = double(tve.tv_sec-tvb.tv_sec)+1.e-9*(double(tve.tv_nsec)-double(tvb.tv_nsec));
    printf("RxRecClk rate = %f MHz\n", 16.e-6*double(vve-vvb)/dt);
  }

  if (lRing0 || lRing1) {
    RingBuffer& b = *new((char*)p->reg()+(lRing0 ? 0x50000 : 0x60000)) RingBuffer;
    b.clear ();
    b.enable(true);
    usleep(100);
    b.enable(false);
    b.dump();
  }

  //  p->dumpPgp();

  return 0;
}

void* debugThread(void* args)
{
  unsigned count=0;
  while(1) {
    usleep(1000);
    printf("\r%u",count);
    count++;
  }
}
