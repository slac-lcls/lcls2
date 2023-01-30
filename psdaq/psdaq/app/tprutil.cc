
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <cinttypes>
#include <time.h>

#include "psdaq/tpr/Module.hh"
#include "psdaq/tpr/Queues.hh"
//#include "CmdLineTools.hh"
#include <pthread.h>

#include <string>

using namespace Pds::Tpr;

static const double CLK_FREQ = 1300e6/7.;
static unsigned linktest_period = 1;
static unsigned triggerPolarity = 1;
static unsigned triggerDelay = 1;
static unsigned triggerWidth = 0;
static bool     lRevMarkers = false;
static bool     verbose = false;

extern int optind;

static void link_test          (TprReg&, bool lcls2, int mode, int n, bool lring);
static void frame_rates        (TprReg&, bool lcls2, int n=1);
static void frame_capture      (TprReg&, char, bool lcls2, unsigned frames);
static void dump_frame         (const uint32_t*);
static bool parse_frame        (const uint32_t*, uint64_t&, uint64_t&, uint16_t&);
static bool parse_bsa_event    (const uint32_t*, uint64_t&, uint64_t&,
                                uint64_t&, uint64_t&, uint64_t&);
static bool parse_bsa_control  (const uint32_t*, uint64_t&, uint64_t&,
                                uint64_t&, uint64_t&, uint64_t&);
static void generate_triggers  (TprReg&, bool lcls2);

static void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("          -d <dev>  : <tpr a/b>\n");
  printf("          -1        : test LCLS-I timing\n");
  printf("          -2        : test LCLS-II timing\n");
  printf("          -m <mode> : force timing protocol (0=LCLS1,1=LCLS2)\n");
  printf("          -n        : skip frame capture test\n");
  printf("          -f <#>    : <n> frames per test (default=10)\n");
  printf("          -N <#>    : <n> frame rate tests\n");
  printf("          -r        : dump ring buffers\n");
  printf("          -v        : verbose\n");
  printf("          -L        : set xbar to loopback\n");
  printf("          -D delay[,width[,polarity]]  : trigger parameters\n");
  printf("          -T <sec>  : link test period\n");
  printf("          -R        : reverse fixed rate markers\n");
}

int main(int argc, char** argv) {

  extern char* optarg;
  char tprid='a';

  int c;
  bool lUsage = false;

  bool lTestI  = false;
  bool lTestII = false;
  bool lFrameTest = true;
  unsigned frames = 10;
  int  nIterations = 1;
  bool loopback   = false;
  bool lDumpRingb = false;
  int mode = -1;
  char* endptr;

  while ( (c=getopt( argc, argv, "12d:f:m:nN:rT:D:LRvh?")) != EOF ) {
    switch(c) {
    case '1': lTestI  = true; break;
    case '2': lTestII = true; break;
    case 'f': frames = strtoul(optarg,NULL,0); break;
    case 'm': mode   = strtoul(optarg,NULL,0); break;
    case 'n': lFrameTest = false; break;
    case 'N': nIterations = strtoul(optarg,NULL,0); break;
    case 'r': lDumpRingb = true; break;
    case 'v': verbose = true; break;
    case 'd':
      tprid  = optarg[0];
      if (strlen(optarg) != 1) {
        printf("%s: option `-r' parsing error\n", argv[0]);
        lUsage = true;
      }
      break;
    case 'D':
      triggerWidth = 1;
      triggerDelay = strtoul(optarg,&endptr,0);
      if (endptr[0]==',')
        triggerWidth = strtoul(endptr+1,&endptr,0);
      if (endptr[0]==',')
        triggerPolarity = strtoul(endptr+1,&endptr,0);
      break;
    case 'L':
      loopback = true;
      break;
    case 'R':
      lRevMarkers = true;
      break;
    case 'T':
      linktest_period = strtoul(optarg,NULL,0);
      break;
    case 'h':
      usage(argv[0]);
      exit(0);
    case '?':
    default:
      lUsage = true;
      break;
    }
  }

  if (optind < argc) {
    printf("%s: invalid argument -- %s\n",argv[0], argv[optind]);
    lUsage = true;
  }

  if (lUsage) {
    usage(argv[0]);
    exit(1);
  }

  {
    TprReg* p = reinterpret_cast<TprReg*>(0);
      printf("version @%p\n",&p->version);
      printf("xbar    @%p\n",&p->xbar);
      printf("base    @%p\n",&p->base);
      printf("tpr     @%p\n",&p->tpr);
      printf("tpg     @%p\n",&p->tpg);
    printf("RxRecClks[%p]\n",&p->tpr.RxRecClks);
  }

  {
    char evrdev[16];
    sprintf(evrdev,"/dev/tpr%c",tprid);
    printf("Using tpr %s\n",evrdev);

    int fd = open(evrdev, O_RDWR);
    if (fd<0) {
      perror("Could not open");
      return -1;
    }

    void* ptr = mmap(0, sizeof(TprReg), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
      perror("Failed to map");
      return -2;
    }

    TprReg& reg = *reinterpret_cast<TprReg*>(ptr);
    printf("FpgaVersion: %08X\n", reg.version.FpgaVersion);
    printf("BuildStamp: %s\n", reg.version.buildStamp().c_str());

    if (loopback) {
        reg.xbar.setEvr( XBar::LoopIn );
        reg.xbar.setEvr( XBar::LoopOut);
        reg.xbar.setTpr( XBar::LoopIn );
        reg.xbar.setTpr( XBar::LoopOut);
    }
    else {
        reg.xbar.setEvr( XBar::StraightIn );
        reg.xbar.setEvr( XBar::StraightOut);
        reg.xbar.setTpr( XBar::StraightIn );
        reg.xbar.setTpr( XBar::StraightOut);
    }

    if (lTestII) {
      //
      //  Validate LCLS-II link
      //
      link_test(reg, true, mode, nIterations, lDumpRingb);

      //
      //  Capture series of timing frames (show table)
      //
      if (lFrameTest) {
          frame_rates  (reg, mode!=0, nIterations);
          frame_capture(reg,tprid, true, frames);
      }
      //
      //  Generate triggers (what device can digitize them)
      //
      if (triggerWidth)
        generate_triggers(reg, mode!=0);
    }

    if (lTestI) {
      //
      //  Validate LCLS-I link
      link_test(reg, false, mode, nIterations, lDumpRingb);

      //
      //  Capture series of timing frames (show table)
      //
      if (lFrameTest) {
        frame_rates  (reg, mode==1, nIterations);
        frame_capture(reg,tprid, false, frames);
      }
      //
      //  Generate triggers
      //
      if (triggerWidth)
        generate_triggers(reg, mode==1);
    }
  }

  return 0;
}

void link_test(TprReg& reg, bool lcls2, int mode, int n, bool lring)
{
  static const double ClkMin[] = { 118, 184 };
  static const double ClkMax[] = { 120, 187 };
  static const double FrameMin[] = { 356, 928000 };
  static const double FrameMax[] = { 362, 929000 };
  unsigned ilcls = lcls2 ? 1:0;

  reg.tpr.clkSel(lcls2);
  reg.tpr.modeSel(mode!=0);
  reg.tpr.modeSelEn(mode>=0);
  reg.tpr.rxPolarity(false);
  do {
      usleep(100000);
      reg.tpr.resetCounts();
      unsigned rxclks0 = reg.tpr.RxRecClks;
      unsigned txclks0 = reg.tpr.TxRefClks;
      usleep(linktest_period*1000000);
      unsigned rxclks1 = reg.tpr.RxRecClks;
      unsigned txclks1 = reg.tpr.TxRefClks;
      unsigned sofCnts = reg.tpr.SOFcounts;
      unsigned crcErrs = reg.tpr.CRCerrors;
      unsigned decErrs = reg.tpr.RxDecErrs;
      unsigned dspErrs = reg.tpr.RxDspErrs;
      double rxClkFreq = double(rxclks1-rxclks0)*16.e-6;
      printf("RxRecClkFreq: %7.2f  %s\n",
             rxClkFreq,
             (rxClkFreq > ClkMin[ilcls] &&
              rxClkFreq < ClkMax[ilcls]) ? "PASS":"FAIL");
      double txClkFreq = double(txclks1-txclks0)*16.e-6;
      printf("TxRefClkFreq: %7.2f  %s\n",
             txClkFreq,
             (txClkFreq > ClkMin[ilcls] &&
              txClkFreq < ClkMax[ilcls]) ? "PASS":"FAIL");
      printf("SOFcounts   : %7u  %s\n",
             sofCnts,
             (sofCnts > FrameMin[ilcls] &&
              sofCnts < FrameMax[ilcls]) ? "PASS":"FAIL");
      printf("CRCerrors   : %7u  %s\n",
             crcErrs,
             crcErrs == 0 ? "PASS":"FAIL");
      printf("DECerrors   : %7u  %s\n",
             decErrs,
             decErrs == 0 ? "PASS":"FAIL");
      printf("DSPerrors   : %7u  %s\n",
             dspErrs,
             dspErrs == 0 ? "PASS":"FAIL");
  } while (--n);

  reg.tpr.dump();
  reg.csr.dump();

  unsigned v = reg.tpr.CSR;
  printf(" %s", v&(1<<1) ? "LinkUp":"LinkDn");
  if (v&(1<<2)) printf(" RXPOL");
  printf(" %s", v&(1<<4) ? "LCLSII":"LCLS");
  if (v&(1<<5)) printf(" LinkDnL");
  printf("\n");
  //  Acknowledge linkDownL bit
  reg.tpr.CSR = v & ~(1<<5);

  if (!lring) return;

  //  Dump ring buffer
  printf("\n-- RingB 0 --\n");
  reg.ring0.enable(false);
  reg.ring0.clear ();
  reg.ring0.enable(true);
  usleep(10000);
  reg.ring0.enable(false);
  reg.ring0.dump  ();

  //  Dump ring buffer
  printf("\n-- RingB 1 --\n");
  reg.ring1.enable(false);
  reg.ring1.clear ();
  reg.ring1.enable(true);
  usleep(10000);
  reg.ring1.enable(false);
  reg.ring1.dump  ("%08x");
}

void frame_rates(TprReg& reg, bool lcls2, int n)
{
  const unsigned nrates=7;
  unsigned ilcls = lcls2 ? nrates : 0;
  unsigned rates[nrates];
  static const unsigned rateMin[] = { 356, 116, 56, 27,  8, 3, 0, 909999, 69999,  9999,  999,  99,  9, 0 };
  static const unsigned rateMax[] = { 364, 124, 64, 33, 12, 7, 2, 910001, 70001, 10001, 1001, 101, 11, 2 };

  //  for(unsigned i=0; i<nrates; i++) {
  for(unsigned i=0; i<TprBase::NCHANNELS; i++) {
      if (lcls2)
          reg.base.channel[i].evtSel  = (1<<30) | (i%nrates);
      else {
          switch(i%nrates) {
          case 0:
              reg.base.channel[i].evtSel  = (1<<30) | (1<<11) | (0x3f<<3);
              break;
          case 1:
              reg.base.channel[i].evtSel  = (1<<30) | (1<<11) | (0x11<<3);
              break;
          default:
              reg.base.channel[i].evtSel  = (1<<30) | (1<<11) | (((i%nrates)-2)<<0) | (0x1<<3);
              break;
          }
      }
      reg.base.channel[i].control = 1;
  }

  do {
      usleep(2000000);

      for(unsigned i=0; i<nrates; i++)
          rates[i] = reg.base.channel[i].evtCount;

      for(unsigned i=0; i<nrates; i++) {
          unsigned rate = rates[i];
          unsigned ridx = lRevMarkers ? 6-i+ilcls : i+ilcls;
          printf("FixedRate[%i]: %7u  %s\n",
                 i, rate,
                 (rate > rateMin[ridx] &&
                  rate < rateMax[ridx]) ? "PASS":"FAIL");
      }
  } while (--n);

  for(unsigned i=0; i<TprBase::NCHANNELS; i++) {
      reg.base.channel[i].control = 0;
  }
}

class ThreadArgs {
public:
    ThreadArgs() {}
    ThreadArgs(int _fd, void* _qptr, int _idx, char _tpr_id, bool _lcls2, unsigned _frames) :
        fd(_fd), qptr(_qptr), idx(_idx), tpr_id(_tpr_id), lcls2(_lcls2), frames(_frames) {}
public:
    int  fd;
    void* qptr;
    int  idx;
    char tpr_id;
    bool lcls2;
    unsigned frames;
};

void* file_monitor_thread(void* a)
{
  ThreadArgs* args = (ThreadArgs*)a;
  void*    ptr     = args->qptr;
  int      tprid   = args->fd;
  int      idx     = args->idx;

  Queues& q = *(Queues*)ptr;

  char dev[16];
  sprintf(dev,"/dev/tpr%c%x",tprid,idx);

  int fd = open(dev, O_RDONLY);
  if (fd<0) {
    printf("Open failure for dev %s [FAIL]\n",dev);
    perror("Could not open");
    return 0;
  }

  int64_t allrp = q.allwp[idx];
  //int64_t bsarp = q.bsawp;

  //  read(fd, buff, 32);

  do {
    printf("idx %x  allrp %" PRIx64 "  q.allwp %l" PRIx64 "  q.allrp.idx %l" PRIx64 "\n",
           idx, allrp, q.allwp[idx], q.allrp[idx].idx[allrp &(MAX_TPR_ALLQ-1)]);
    allrp = q.allwp[idx];
    //bsarp = q.bsawp;
    usleep(1000000);
    //    read(fd, buff, 32);
  } while(1);

  return 0;
}

void* frame_capture_thread(void* a)
{
  ThreadArgs* args = (ThreadArgs*)a;
  int      fd      = args->fd;
  void*    ptr     = args->qptr;
  int      idx     = args->idx;
  bool     lcls2   = args->lcls2;
  unsigned frames  = args->frames;

  Queues& q = *(Queues*)ptr;

  char* buff = new char[32];

  int64_t allrp = q.allwp[idx];
  //int64_t bsarp = q.bsawp;

  read(fd, buff, 32);
  //  usleep(lcls2 ? 20 : 100000);
  //  usleep(100000);
  //  disable channel 0
  //  reg.base.channel[_channel].control = ucontrol;

  uint64_t pulseIdP=0;
  uint64_t pulseId, timeStamp;
  uint16_t markers;
  unsigned nframes=0;

  const unsigned _1HzM = lRevMarkers ? 0x1 : 0x40;

  do {
    while(allrp < q.allwp[idx] && (nframes<frames || frames==0)) {
      const uint32_t* p = reinterpret_cast<const uint32_t*>
        (&q.allq[q.allrp[idx].idx[allrp &(MAX_TPR_ALLQ-1)] &(MAX_TPR_ALLQ-1) ].word[0]);
      if (verbose) {
          printf("allrp %" PRIx64 "  q.allwp %l" PRIx64 "  q.allrp.idx %l" PRIx64 "\n",
                 allrp, q.allwp[idx], q.allrp[idx].idx[allrp &(MAX_TPR_ALLQ-1)]);
        dump_frame(p);
      }
      if (parse_frame(p, pulseId, timeStamp, markers)) {
        if (pulseIdP) {
            //  EvrCardG2 replaces PulseId with Timestamp
            //  Timestamp increments at 1MHz and jumps at next fiducial
          uint64_t pulseIdN = pulseIdP+1;
          if (!lcls2) pulseIdN = (pulseId&~0x1ffffULL) | (pulseIdN&0x1ffffULL);
          if (frames) {
              printf(" 0x%04x  0x%016llx %9u.%09u %s\n",
                     p[0],
                     (unsigned long long)pulseId,
                     unsigned(timeStamp>>32),
                     unsigned(timeStamp&0xffffffff),
                     (pulseId==pulseIdN) ? "PASS":"FAIL");
          }
          else if (pulseId!=pulseIdN && ((markers&0x400)==0)) {  // 60Hz marker makes TS jump
              printf(" 0x%016llx %9u.%09u %" PRId64 "\n",
                     (unsigned long long)pulseId,
                     unsigned(timeStamp>>32),
                     unsigned(timeStamp&0xffffffff),
                     (pulseId-pulseIdN));
          }
          else if ((markers&_1HzM)==_1HzM) {
              printf(" 0x%016llx %9u.%09u -\n",
                     (unsigned long long)pulseId,
                     unsigned(timeStamp>>32),
                     unsigned(timeStamp&0xffffffff));
          }
          nframes++;
        }
        pulseIdP  =pulseId;
      }
      allrp++;
    }
    if (nframes>=frames && frames!=0)
      break;
    usleep(10000);
    read(fd, buff, 32);
  } while(1);

  return 0;
}

void frame_capture(TprReg& reg, char tprid, bool lcls2, unsigned frames)
{
  unsigned idx=13;
  char dev[16];
  sprintf(dev,"/dev/tpr%c%x",tprid,idx);

  int fd = open(dev, O_RDONLY);
  if (fd<0) {
    printf("Open failure for dev %s [FAIL]\n",dev);
    perror("Could not open");
    return;
  }

  char bsadev[16];
  sprintf(bsadev,"/dev/tpr%cBSA",tprid);
  int fdbsa = open(bsadev, O_RDWR);
  if (fdbsa<0) {
      perror("Could not open");
      return;
  }


  void* ptr = mmap(0, sizeof(Queues), PROT_READ, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    perror("Failed to map - FAIL");
    return;
  }

  unsigned _channel = idx;
  unsigned ucontrol = reg.base.channel[_channel].control;
  reg.base.channel[_channel].control = 0;

  reg.base.dump();

  unsigned urate   = lcls2 ? (lRevMarkers ? 6:0) : (1<<11) | (0x3f<<3); // max rate
  unsigned destsel = 1<<17; // BEAM - DONT CARE
  reg.base.channel[_channel].evtSel = (destsel<<13) | (urate<<0);
  reg.base.channel[_channel].bsaDelay = 0;
  reg.base.channel[_channel].bsaWidth = 1;
  reg.base.channel[_channel].control = ucontrol | 1;

  ThreadArgs hargs[14];
  for(unsigned i=0; i<14; i++) {
    if (i==idx) continue;
    new((void*)&hargs[i]) ThreadArgs(tprid, ptr, i, tprid, lcls2, frames);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_t      thread_id;
    pthread_create(&thread_id, &attr, &file_monitor_thread, (void*)&hargs[i]);
  }

  //  read the captured frames
  printf("   %16.16s %8.8s %8.8s\n",
         "PulseId","Seconds","Nanosec");

  ThreadArgs args(fd, ptr, idx, tprid, lcls2, frames);
  frame_capture_thread(&args);

  Queues& q = *(Queues*)ptr;

  char* buff = new char[32];

  //int64_t allrp = q.allwp[idx];
  int64_t bsarp = q.bsawp;

  read(fdbsa, buff, 32);
  //  usleep(lcls2 ? 20 : 100000);
  //  usleep(100000);
  //  disable channel 0
  //  reg.base.channel[_channel].control = ucontrol;

  //uint64_t pulseIdP=0;
  uint64_t pulseId, timeStamp;
  //uint16_t markers;
  unsigned nframes=0;

  printf("BSA frames\n");

  uint64_t active, avgdn, update, init, minor, major;
  nframes = 0;
  do {
    while(bsarp < q.bsawp && nframes<frames) {
      const uint32_t* p = reinterpret_cast<const uint32_t*>
        (&q.bsaq[bsarp &(MAX_TPR_BSAQ-1)].word[0]);
      if (parse_bsa_control(p, pulseId, timeStamp, init, minor, major)) {
        printf(" 0x%016llx %9u.%09u I%016llx m%016llx M%016llx\n",
               (unsigned long long)pulseId,
               unsigned(timeStamp>>32),
               unsigned(timeStamp&0xffffffff),
               (unsigned long long)init,
               (unsigned long long)minor,
               (unsigned long long)major);
      }
      if (parse_bsa_event(p, pulseId, timeStamp, active, avgdn, update)) {
        printf(" 0x%016llx %9u.%09u A%016llx D%016llx U%016llx\n",
               (unsigned long long)pulseId,
               unsigned(timeStamp>>32),
               unsigned(timeStamp&0xffffffff),
               (unsigned long long)active,
               (unsigned long long)avgdn,
               (unsigned long long)update);
        nframes++;
      }
      bsarp++;
    }
    if (nframes>=frames)
      break;
    usleep(10000);
    read(fdbsa, buff, 32);
  } while(1);

  //  disable channel 0
  reg.base.channel[_channel].control = ucontrol;

  munmap(ptr, sizeof(Queues));
  close(fd);
}

void dump_frame(const uint32_t* p)
{
  char m = p[0]&(0x808<<20) ? 'D':' ';
  if (((p[0]>>16)&0xf)==0) {
    const uint64_t* pl = reinterpret_cast<const uint64_t*>(p+2);
    printf("EVENT LCLS%c chmask [x%x] [x%x] %c: %16lx %16lx",
           (p[0]&(1<<22)) ? '1':'2',
           (p[0]>>0)&0xffff,p[1],m,pl[0],pl[1]);
    for(unsigned i=6; i<20; i++)
      printf(" %08x",p[i]);
    printf("\n");
  }
}

bool parse_frame(const uint32_t* p,
                 uint64_t& pulseId, uint64_t& timeStamp, uint16_t& markers)
{
  //  char m = p[0]&(0x808<<20) ? 'D':' ';
  if (((p[0]>>16)&0xf)==0) { // EVENT_TAG
    const uint64_t* pl = reinterpret_cast<const uint64_t*>(p+2);
    pulseId   = pl[0];
    timeStamp = pl[1];
    markers   = pl[2]&0xffff;
    return true;
  }
  return false;
}

bool parse_bsa_event(const uint32_t* p,
                     uint64_t& pulseId, uint64_t& timeStamp,
                     uint64_t& active, uint64_t& avgdone, uint64_t& update)
{
  if (((p[0]>>16)&0xf)==2) { // BSAEVNT_TAG
    const uint64_t* pl = reinterpret_cast<const uint64_t*>(p+1);
    pulseId   = pl[0];
    active    = pl[1];
    avgdone   = pl[2];
    timeStamp = pl[3];
    return true;
  }
  return false;
}

bool parse_bsa_control(const uint32_t* p,
                       uint64_t& pulseId, uint64_t& timeStamp,
                       uint64_t& init, uint64_t& minor, uint64_t& major)
{
  if (((p[0]>>16)&0xf)==1) { // BSACNTL_TAG
    const uint64_t* pl = reinterpret_cast<const uint64_t*>(p+1);
    pulseId   = pl[0];
    timeStamp = pl[1];
    init      = pl[2];
    minor     = pl[3];
    major     = pl[4];
    return true;
  }
  return false;
}

void generate_triggers(TprReg& reg, bool lcls2)
{
  unsigned _channel = 0;
  reg.base.setupTrigger(0,_channel,triggerPolarity,0,triggerWidth);
  for(unsigned i=1; i<12; i++)
    reg.base.setupTrigger(i,
                          _channel,
                          triggerPolarity, triggerDelay, triggerWidth+i, 0); // polarity, delay, width, tap

  unsigned ucontrol = 0;
  reg.base.channel[_channel].control = ucontrol;

  unsigned urate   = lcls2 ? (lRevMarkers ? 6:0) : (1<<11) | (0x3f<<3); // max rate
  //unsigned urate   = (1<<11) | (0x9<<3); // max rate
  unsigned destsel = 1<<17; // BEAM - DONT CARE
  reg.base.channel[_channel].evtSel = (destsel<<13) | (urate<<0);
  reg.base.channel[_channel].bsaDelay = 0;
  reg.base.channel[_channel].bsaWidth = 1;
  reg.base.channel[_channel].control = ucontrol | 1;

  reg.base.dump();
}
