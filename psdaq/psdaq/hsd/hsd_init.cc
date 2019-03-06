
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

#include "psdaq/hsd/Module.hh"
#include "psdaq/hsd/Globals.hh"
#include "psdaq/hsd/TprCore.hh"
#include "psdaq/hsd/QABase.hh"
#include "psdaq/mmhw/RingBuffer.hh"

using Pds::Mmhw::RingBuffer;

#include <string>

extern int optind;

using namespace Pds::HSD;

static double calc_phase(unsigned even,
                         unsigned odd)
{
  const double periodr = 1000./156.25;
  const double periodt = 7000./1300.;
  if (even)
    return double(even)/double(0x80000)*periodt;
  else
    return double(odd )/double(0x80000)*periodt + 0.5*periodr;
}

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
  printf("\t-T <train i/o delays>\n");
  printf("\t-W filename <write new PROM>\n");
  printf("\t-0 <dump raw timing receive buffer>\n");
  printf("\t-1 <dump timing message buffer>\n");
  printf("\t-2 <configure for LCLSII>\n");
  printf("\t-3 <configure for EXTERNAL>\n");
  printf("\t-4 <configure for K929>\n");
  printf("\t-5 <configure for M3_7>\n");
  printf("\t-6 <configure for M7_4>\n");
  //  printf("Options: -a <IP addr (dotted notation)> : Use network <IP>\n");
}

void* debugThread(void*);

int main(int argc, char** argv) {

  extern char* optarg;
  char* endptr;

  const char* devName = "/dev/qadca";
  int c;
  bool lUsage = false;
  bool lSetupClkSynth = false;
  bool lReset = false;
  bool lResetRx = false;
  bool lResetTx = false;
  bool lPolarity = false;
  bool lRing0 = false;
  bool lRing1 = false;
  bool lTrain = false;
  bool lTrainNoReset = false;
  bool lClkSync = false;
  bool lAdcSync = false;
  bool lAdcSyncRst = false;
  TimingType timing=LCLS;

  const char* fWrite=0;
#if 0
  bool lSetPhase = false;
  unsigned delay_int=0, delay_frac=0;
#endif
  unsigned trainRefDelay = 0;

  while ( (c=getopt( argc, argv, "CDRXYP0123456d:htST:UW:Z")) != EOF ) {
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
    case '2':
      timing = LCLSII;
      break;
    case '3':
      timing = EXTERNAL;
      break;
    case '4':
      timing = K929;
      break;
    case '5':
      timing = M3_7;
      break;
    case '6':
      timing = M7_4;
      break;
    case 'd':
      devName = optarg;
      break;
    case 't':
      lTrainNoReset = true;
      break;
    case 'T':
      lTrain = true;
      trainRefDelay = strtoul(optarg,&endptr,0);
      break;
    case 'U':
      lClkSync = true;
      break;
    case 'W':
      fWrite = optarg;
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

  int fd = open(devName, O_RDWR);
  if (fd<0) {
    perror("Open device failed");
    return -1;
  }

  Module* p = Module::create(fd);

  p->dumpMap();

  p->board_status();

  p->fmc_dump();

  { unsigned trg_even = p->trgPhase()[0];
    unsigned trg_odd  = p->trgPhase()[1];
    unsigned clkt_even = p->trgPhase()[2];
    unsigned clkt_odd  = p->trgPhase()[3];
    double trg_ph  = calc_phase(trg_even,trg_odd);
    double clkt_ph = calc_phase(clkt_even,clkt_odd);
    printf("Trigger Phase: %x/%x %x/%x [%f %f]\n",
           trg_even, trg_odd,
           clkt_even, clkt_odd,
           trg_ph, clkt_ph);
  }

  if (lSetupClkSynth) {
    p->fmc_clksynth_setup(timing);
  }

  if (lPolarity) {
    p->tpr().rxPolarity(!p->tpr().rxPolarity());
  }

  if (lResetRx) {
    if (lSetupClkSynth)
      sleep(1);

    switch(timing) {
    case LCLS:
      p->tpr().setLCLS();
      break;
    case LCLSII:
      p->tpr().setLCLSII();
      break;
    default:
      return 0;
      break;
    }
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
    p->clocktree_sync();
  }

  if (lAdcSync) {
    p->sync();
  }

  if (lAdcSyncRst) {
    QABase& base = *reinterpret_cast<QABase*>((char*)p->reg()+0x80000);
    base.resetClock(true);
  }

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

  if (lTrain) {
    if (lResetRx)
      sleep(1);
    p->fmc_init(timing);
    p->train_io(trainRefDelay);
  }

  if (lTrainNoReset) {
    p->train_io(trainRefDelay);
  }

  if (fWrite)
    p->flash_write(fWrite);

  if (lRing0 || lRing1) {
    RingBuffer& b = *new((char*)p->reg()+(lRing0 ? 0x50000 : 0x60000)) RingBuffer;
    b.clear ();
    b.enable(true);
    usleep(100);
    b.enable(false);
    b.dump();
  }

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
