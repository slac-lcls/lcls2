#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <stdlib.h>
#include <signal.h>
#include <argp.h>
#include "DataDriver.h"
using namespace std;

#define EVENT_COUNT_ERR   0x01
#define FRAME_COUNT_ERR   0x02
#define FRAME_CONTENT_ERR 0x04
#define FRAME_SIZE_ERR    0x08

#define HISTORY_ERR       0x10
#define LANE_MISS_ERR     0x20
#define PULSE_ID_ERR      0x40
#define FRAME_COUNT_MISM  0x80

#define PRINT_ANY         0x80000000

static unsigned verbose = 0;
static FILE* fdump = 0;
static bool _validateFrameContent = true;
static unsigned bufferIndex = 0;

static void usage(const char* p) {
  printf("Usage: %s [options'\n",p);
  printf("Options:\n");
  printf("\t-d <device file>     (default: /dev/datadev_1)\n");
  printf("\t-f <dump filename>   (default: none)\n");
  printf("\t-c <update interval> (default: 1000000)\n");
  printf("\t-C partition[,length[,links]] [configure simcam]\n");
  printf("\t-F                   [reset frame counters]\n");
  printf("\t-N                   [dont validate frame contents]\n");
  printf("\t-v <verbose mask>    (default: 0)\n");
  printf("\t                     (bit 0 : event counter not incr by 1)\n");
  printf("\t                     (bit 1 : frame counter not incr by 1)\n");
  printf("\t                     (bit 2 : frame content error)\n");
  printf("\t                     (bit 3 : frame size error)\n");
  printf("\t                     (bit 4 : lane contr more than once)\n");
  printf("\t                     (bit 5 : lane missing)\n");
  printf("\t                     (bit 6 : pulse ID mismatch)\n");
  printf("\t                     (bit 7 : frame counter mismatch)\n");
}

#define HISTORY (32*1024)
#define MAX_PRINT 32

static unsigned nprint = 0;

class LaneValidator {
public:
  LaneValidator() : _current(1<<31),
                    _sz     (0),
                    _ncalls (0),
                    _eventCounterErr(0),
                    _frameCounterErr(0),
                    _frameContentErr(0),
                    _frameSizeErr   (0),
                    _reventCounterErr(0),
                    _rframeCounterErr(0),
                    _rframeContentErr(0),
                    _rframeSizeErr   (0) {}
public:
  void validate(const uint32_t* p, unsigned sz);
  void report() {
#define RERR(s) \
    if (_##s != _r##s) {                \
      printf(#s " : %u\n", _##s-_r##s); \
      _r##s=_##s;                       \
    }
    RERR(eventCounterErr);
    RERR(frameCounterErr);
    RERR(frameContentErr);
    RERR(frameSizeErr);
#undef RERR
  }
  void summary() {
#define RERR(s) \
    if (_##s != 0) {                    \
      printf(#s " : %u\n", _##s);       \
    }
    RERR(ncalls);
    RERR(eventCounterErr);
    RERR(frameCounterErr);
    RERR(frameContentErr);
    RERR(frameSizeErr);
#undef RERR
  }
public:
  uint64_t _pulseId [HISTORY];
  unsigned _frameCnt[HISTORY];
  unsigned _current; // event counter
  unsigned _sz;
  unsigned _ncalls;
  unsigned _eventCounterErr;
  unsigned _frameCounterErr;
  unsigned _frameContentErr;
  unsigned _frameSizeErr;
  unsigned _reventCounterErr;
  unsigned _rframeCounterErr;
  unsigned _rframeContentErr;
  unsigned _rframeSizeErr;
};

class EventValidator {
public:
  EventValidator(const LaneValidator* valid,
                 unsigned             lanemask) : 
    _lanev   (valid),
    _lanemask(lanemask),
    _last    (1<<31),
    _ncalls     (0),
    _laneMissErr(0),
    _pulseIdErr (0),
    _frameCntErr(0),
    _rlaneMissErr(0),
    _rpulseIdErr (0),
    _rframeCntErr(0)
  { memset(_lanes,0,HISTORY); }
public:
  void validate(const uint32_t* p, unsigned sz, unsigned lane) {
#define DUMP(s) {                                                       \
      printf("\t" #s " :");                                             \
      for(unsigned k=0; k<4; k++) {                                     \
        unsigned long long v = _lanev[k]._##s[now];                     \
          printf(" %llx", v);                                           \
      }                                                                 \
      printf("\n");                                                     \
    }
    unsigned event = p[5]&0xffffff;
    unsigned now = event % HISTORY;
    unsigned pre = (event-1) % HISTORY;
    _ncalls++;

    if (_lanes[now] & (1<<lane)) {
      _historyErr++;
      if ((verbose & HISTORY_ERR) && nprint++ < MAX_PRINT)
        printf("\thistory : %x [%x]\n", _lanes[now], 1<<lane);
    }

    if ((_lanes[now]|=(1<<lane)) == _lanemask) {
      _last = now;
      if (_lanes[pre]) {
        _laneMissErr++;
        if ((verbose & LANE_MISS_ERR) && nprint++ < MAX_PRINT) {
          printf("lanemiss[%x.%x]:",pre,_lanes[pre]);
          for(unsigned i=0; i<4; i++)
            printf(" %06x",_lanev[i]._current);
          printf("\n");
        }
        _lanes[pre] = 0;  // incomplete reported, now clear
      }
      uint64_t pid  = _lanev[0]._pulseId[now];
      unsigned lm  = _lanemask & ~1;
      for(unsigned i=1; lm; i++) {
        lm &= ~(1<<i);
        if (_lanev[i]._pulseId[now] != pid) {
          _pulseIdErr++;
          if ((verbose & PULSE_ID_ERR) && nprint++ < MAX_PRINT) 
            DUMP(pulseId);
          break;
        }
      }
      if (p[4]>>31) { // L1Accept
        unsigned fcnt = _lanev[0]._frameCnt[now];
        lm  = _lanemask & ~1;
        for(unsigned i=1; lm; i++) {
          lm &= ~(1<<i);
          if (_lanev[i]._frameCnt[now] != fcnt) {
            _frameCntErr++;
            if ((verbose & FRAME_COUNT_MISM) && nprint++ < MAX_PRINT)
              DUMP(frameCnt);
            break;
          }
        }
      }
      _lanes[now] = 0;
    }
  }
  void report() {
#if 0
    printf("\t");
    unsigned cmn=(1<<24);
    unsigned cmx=0;
    for(unsigned i=0; i<4; i++) {
      unsigned v = _lanev[i]._current;
      printf(" %06x", v);
      if (v > cmx) cmx=v;
      if (v < cmn) cmn=v;
    }
    printf(" [%x]\n", cmx-cmn);
#endif
#define RERR(s)                                 \
    if (_##s != _r##s) {                        \
      printf(#s " : %u\n", _##s-_r##s);         \
      _r##s=_##s;                               \
    }
    RERR(historyErr);
    RERR(laneMissErr);
    RERR(pulseIdErr);
    RERR(frameCntErr);
#undef RERR
    //    nprint = 0;
  }
  void summary() {
#define RERR(s)                                 \
    if (_##s != 0) {                            \
      printf(#s " : %u\n", _##s);               \
    }
    RERR(ncalls);
    RERR(historyErr);
    RERR(laneMissErr);
    RERR(pulseIdErr);
    RERR(frameCntErr);
#undef RERR
  }
public:
  const LaneValidator* _lanev;
  uint8_t  _lanes  [HISTORY];
  unsigned _lanemask;
  unsigned _last;  // event counter of last complete
  unsigned _ncalls;
  unsigned _historyErr;
  unsigned _laneMissErr;
  unsigned _pulseIdErr;
  unsigned _frameCntErr;
  unsigned _rhistoryErr;
  unsigned _rlaneMissErr;
  unsigned _rpulseIdErr;
  unsigned _rframeCntErr;
};

//  Setup the validators
LaneValidator  lanev[4];
EventValidator* eventv;

void LaneValidator::validate(const uint32_t* p, unsigned sz) {
  _ncalls++;
  unsigned event = p[5]&0xffffff;

  if ((p[4]>>31)==0) {  // Transition
    if ((_current & (1<<31))==0) {
      unsigned pre = _current%HISTORY;
      _current = event;                          // account for incrementing event counter
      _frameCnt[event%HISTORY] = _frameCnt[pre]; // account for non-incrementing frame counter
    }
    return;
  }

  //  validate this event
  unsigned now = event%HISTORY;
  _pulseId [now] = *reinterpret_cast<const uint64_t*>(p);
  _frameCnt[now] = p[8];
  if (_validateFrameContent) {
    for(unsigned i=1; i<sz-9; i++)
      if (p[sz-i]!=i) {
        _frameContentErr++;
        if ((verbose & FRAME_CONTENT_ERR) && nprint++ < MAX_PRINT)
          printf("\t[%p] [%06x.%ld]: p[%d] %x\n", p, event, this-lanev, i, p[sz-i]);
        if (fdump) {
          fwrite(&sz,sizeof(unsigned),1,fdump);
          fwrite(p  ,sizeof(unsigned),sz,fdump);
        }
        break;
      }
  }

  if (_current & (1<<31)) { // first event
    _current = event;
    _sz      = sz;
  }
  else {
    unsigned pre = _current%HISTORY;
    _current = (_current+1)&0xffffff;
    if (_current != event) {
      _eventCounterErr++;
      if ((verbose & EVENT_COUNT_ERR) && nprint++ < MAX_PRINT)
        printf("\t[%p] [%06x.%ld] eventCount %06x [%06x]\n",
               p, event, this-lanev, event,_current);
      _current = event;
    }
    if (p[8]!=_frameCnt[pre]+1) {
      _frameCounterErr++;
      if ((verbose & FRAME_COUNT_ERR) && nprint++ < MAX_PRINT) {
        printf("\t[%p] [%06x.%ld] frame %08x [%08x]\n",
               p, event, this-lanev, p[8],_frameCnt[pre]+1);
      }
    }
    if (_sz != sz) {
      _frameSizeErr++;
      if ((verbose & FRAME_SIZE_ERR) && nprint++ < MAX_PRINT)
        printf("\t[%p] [%06x.%ld] frameSize : %x [%x]\n", 
               p, event, this-lanev, sz, _sz);
    }
  }
}

static int    _fd;
static void** _dmaBuffers;

static void dump_status();

void sigHandler( int signal ) {
  psignal( signal, "Signal received by pgpWidget");
  eventv->summary();
  for(unsigned i=0; i<4; i++)
    lanev[i].summary();
  dump_status();
  dmaUnMapDma(_fd,_dmaBuffers);
  ::exit(signal);
}


int main (int argc, char **argv) {
   unsigned count = 1000000;
   unsigned max_ret_cnt = 70000;
   const char* dev  = "/dev/datadev_1";
   const char* dump = 0;
   int partition  = -1;
   int length     = 320;
   int links      = 0xff;
   bool frameRst  = false;
   extern char* optarg;
   char* endptr;
   int c;
   while((c=getopt(argc,argv,"d:f:Fc:C:m:v:N"))!=EOF) {
     switch(c) {
     case 'd': dev     = optarg; break;
     case 'f': dump    = optarg; break;
     case 'm': max_ret_cnt = strtoul(optarg,NULL,0); break;
     case 'F': frameRst = true; break;
     case 'c': count   = strtoul(optarg,NULL,0); break;
     case 'v': verbose = strtoul(optarg,NULL,0); break;
     case 'N': _validateFrameContent=false; break;
     case 'C': partition = strtoul(optarg,&endptr,0);
       if (*endptr==',') {
         length = strtoul(endptr+1,&endptr,0);
         if (*endptr==',')
           links = strtoul(endptr+1,NULL,0);
       }
       break;
     default: usage(argv[0]); return 1;
     }
   }

   int32_t       s;
   int32_t       ret;
   uint8_t*      mask    = new uint8_t [DMA_MASK_SIZE];
   uint32_t*     rxFlags = new uint32_t[max_ret_cnt];
   uint32_t*     dest    = new uint32_t[max_ret_cnt];
   void **       dmaBuffers;
   uint32_t      dmaSize;
   uint32_t      dmaCount;
   uint32_t*     dmaIndex = new uint32_t[max_ret_cnt];
   int32_t*      dmaRet   = new int32_t [max_ret_cnt];
   int32_t       x;
   float         last;
   float         rate;
   float         bw;
   float         duration;
   int32_t       max;
   int32_t       total;

   uint32_t      getCnt = max_ret_cnt;

   struct timeval sTime;
   struct timeval eTime;
   struct timeval dTime;
   struct timeval pTime[7];

   if (dump) {
     fdump = fopen(dump,"w");
   }

   printf("  maxCnt           size      count   duration       rate         bw     Read uS   Return uS\n");

   dmaInitMaskBytes(mask);
   for(unsigned i=0; i<4; i++)
     dmaAddMaskBytes((uint8_t*)mask,dmaDest(i,0));

   if ( (s = open(dev, O_RDWR)) <= 0 ) {
      printf("Error opening %s\n",dev);
      return(1);
   }

#if 1
   if ( (dmaBuffers = dmaMapDma(s,&dmaCount,&dmaSize)) == NULL ) {
      perror("Failed to map dma buffers!");
      return(0);
   }
#endif

   _fd         = s;
   _dmaBuffers = dmaBuffers;

   eventv = new EventValidator(lanev,links);

   ::signal( SIGINT, sigHandler );

   dmaSetMaskBytes(s,mask);

   //  Configure the simulated camera
    if (frameRst) {
      unsigned v; dmaReadRegister(s,0x00a00000,&v);
      unsigned w = v;
      w &= ~(0xf<<28);    // disable and drain
      dmaWriteRegister(s,0x00a00000,w);
      usleep(1000);
      w |=  (1<<3);       // reset
      dmaWriteRegister(s, 0x00a00000,w);
      usleep(1);         
      dmaWriteRegister(s, 0x00a00000,v);
    }

   if (partition >= 0) {
     unsigned v = ((partition&0xf)<<0) |
       ((length&0xffffff)<<4) |
       (links<<28);
     dmaWriteRegister(s,0x00a00000, v);
     unsigned w;
     dmaReadRegister(s,0x00a00000,&w);
     printf("Configured partition [%u], length [%u], links [%x]: [%x](%x)\n",
            partition, length, links, v, w);
     for(unsigned i=0; i<4; i++)
       if (links&(1<<i))
         dmaWriteRegister(s,0x00800084+32*i, 0x1f00);
   }


   while(1) {

      bw     = 0.0;
      rate   = 0.0;
      last   = 0.0;
      max    = 0;
      total  = 0;
      gettimeofday(&sTime,NULL);

      while ( rate < count ) {

         // DMA Read
         gettimeofday(&(pTime[0]),NULL);
         ret = dmaReadBulkIndex(s,getCnt,dmaRet,dmaIndex,rxFlags,NULL,dest);  // 24 usec
         gettimeofday(&(pTime[1]),NULL);

         for (x=0; x < ret; ++x) {
            if ( (last = dmaRet[x]) > 0.0 ) {
               rate += 1.0;
               bw += (last * 8.0);
               const uint32_t* b = reinterpret_cast<const uint32_t*>(dmaBuffers[dmaIndex[x]]);
               unsigned lane = (dest[x]>>8)&7;
               unsigned words = unsigned(last)>>2;
               bufferIndex = dmaIndex[x];

               if ((verbose & PRINT_ANY)) {
                 for(unsigned i=0; i<words; i++)
                   printf("%08x%c",b[i], (i&0xf)==0xf ? '\n':' ');
                 printf("\n");
               }
               else if ((b[4]>>31)==0) { // Print transitions
                 printf("lane%x ret%x buff%x fl%x",
                        lane, dmaRet[x], bufferIndex, rxFlags[x]);
                 for(unsigned i=0; i<9; i++)
                   printf(" %08x",b[i]);
                 printf("\n");
               }

               lanev[lane].validate(b,words);
               eventv    ->validate(b,words,lane);

               //  Print out pulseId/timeStamp to check synchronization across nodes
               if ((b[5]&0xfffff)==0) {
                 printf("event[%06x]:",b[5]&0xffffff);
                 for(unsigned i=0; i<9; i++)
                   printf(" %08x",b[i]);
                 printf("\n");
               }
            }
         }

         gettimeofday(&(pTime[2]),NULL);
         if ( ret > 0 ) dmaRetIndexes(s,ret,dmaIndex);  // 721 usec
         gettimeofday(&(pTime[3]),NULL);

	 if ( total == 0 ) if ( ret > max ) max = ret;
	 total += ret; // 0 usec
      }

      gettimeofday(&eTime,NULL);

      timersub(&eTime,&sTime,&dTime);
      duration = dTime.tv_sec + (float)dTime.tv_usec/1000000.0;

      rate = rate / duration;
      bw   = bw   / duration;

      printf("%8i      %1.3e   %8i   %1.2e   %1.2e   %1.2e    %8li    %8li     \n",max,last,count,duration,rate,bw,
         (pTime[1].tv_usec-pTime[0].tv_usec), (pTime[3].tv_usec-pTime[2].tv_usec));
      eventv->report();
      for(unsigned i=0; i<4; i++)
        lanev[i].report();

      rate = 0.0;
      bw   = 0.0;
   }

   return(0);
}

#define CLIENTS(i)       (0x00800080 + i*0x20)
#define DMA_LANES(i)     (0x00800100 + i*0x20)

static inline uint32_t get_reg32(int reg) {
  unsigned v;
  dmaReadRegister(_fd, reg, &v);
  return v;
}

static void print_dma_lane(const char* name, int addr, int offset, int mask)
{
    printf("%20.20s", name);
    for(int i=0; i<4; i++) {
        uint32_t reg = get_reg32( DMA_LANES(i) + addr);
        printf(" %8x", (reg >> offset) & mask);
    }
    printf("\n");
}

static void print_mig_lane(const char* name, int addr, int offset, int mask)
{
    const unsigned MIG_LANES = 0x00800080;
    printf("%20.20s", name);
    for(int i=0; i<4; i++) {
      uint32_t reg = get_reg32( MIG_LANES + i*32 + addr);
      printf(" %8x", (reg >> offset) & mask);
    }
    printf("\n");
}

static void print_clk_rate(const char* name, int addr) 
{
    const unsigned CLK_BASE = 0x00800100;
    printf("%20.20s", name);
    uint32_t reg = get_reg32( CLK_BASE + addr);
    printf(" %f MHz", double(reg&0x1fffffff)*1.e-6);
    if ((reg>>29)&1) printf(" [slow]");
    if ((reg>>30)&1) printf(" [fast]");
    if ((reg>>31)&1) printf(" [locked]");
    printf("\n");
}

static void print_field(const char* name, int addr, int offset, int mask)
{
    printf("%20.20s", name);
    uint32_t reg = get_reg32( addr);
    printf(" %8x", (reg >> offset) & mask);
    printf("\n");
}

static void print_word (const char* name, int addr) { print_field(name,addr,0,0xffffffff); }

static void print_lane(const char* name, int addr, int offset, int stride, int mask)
{
    printf("%20.20s", name);
    for(int i=0; i<4; i++) {
        uint32_t reg = get_reg32( addr+stride*i);
        printf(" %8x", (reg >> offset) & mask);
    }
    printf("\n");
}


void dump_status()
{
  uint32_t lanes = 4;
  printf("  lanes             :  %u\n", lanes);

  printf("  monEnable         :  %u\n", get_reg32( 0x00800000)&1);

  printf("\n-- migLane Registers --\n");
  print_mig_lane("blockSize  ", 0, 0, 0x1f);
  print_mig_lane("blocksPause", 4, 8, 0x3ff);
  print_mig_lane("blocksFree ", 8, 0, 0x1ff);
  print_mig_lane("blocksQued ", 8,12, 0x1ff);
  print_mig_lane("writeQueCnt",12, 0, 0xff);
  print_mig_lane("wrIndex    ",16, 0, 0x1ff);
  print_mig_lane("wcIndex    ",20, 0, 0x1ff);
  print_mig_lane("rdIndex    ",24, 0, 0x1ff);

  print_clk_rate("axilOther  ",0);
  print_clk_rate("timingRef  ",4);
  print_clk_rate("migA       ",8);
  print_clk_rate("migB       ",12);

  // TDetSemi
  print_field("partition", 0x00a00000,  0, 0xf);
  print_field("length"   , 0x00a00000,  4, 0xffffff);
  print_field("enable"   , 0x00a00000, 28, 0xf);
  print_field("localid"  , 0x00a00004,  0, 0xffffffff);
  print_field("remoteid" , 0x00a00008,  0, 0xffffffff);

  print_lane("cntL0"      , 0x00a00010,  0, 16, 0xffffff);
  print_lane("cntOF"      , 0x00a00010, 24, 16, 0xff);
  print_lane("cntL1A"     , 0x00a00014,  0, 16, 0xffffff);
  print_lane("cntL1R"     , 0x00a00018,  0, 16, 0xffffff);
  print_lane("cntWrFifo"  , 0x00a0001c,  0, 16, 0xff);
  print_lane("cntRdFifo"  , 0x00a0001c,  8, 16, 0xff);
  print_lane("cntMsgDelay", 0x00a0001c, 16, 16, 0xffff);
  print_lane("fullToTrig" , 0x00a00050,  0,  4, 0xfff);
  print_lane("nfullToTrig", 0x00a00050, 16,  4, 0xfff);
  print_lane("txLocked"   , 0x00a00050, 28,  4, 0x1);
  print_lane("resetDone"  , 0x00a00050, 29,  4, 0x1);
  print_lane("buffByDone" , 0x00a00050, 30,  4, 0x1);
  print_lane("buffByErr"  , 0x00a00050, 31,  4, 0x1);

  // TDetTiming
  print_word("SOFcounts" , 0x00c00000);
  print_word("EOFcounts" , 0x00c00004);
  print_word("Msgcounts" , 0x00c00008);
  print_word("CRCerrors" , 0x00c0000c);
  print_word("RxRecClks" , 0x00c00010);
  print_word("RxRstDone" , 0x00c00014);
  print_word("RxDecErrs" , 0x00c00018);
  print_word("RxDspErrs" , 0x00c0001c);
  print_word("CSR"       , 0x00c00020);
  print_field("  linkUp" , 0x00c00020, 1, 1);
  print_field("  polar"  , 0x00c00020, 2, 1);
  print_field("  clksel" , 0x00c00020, 4, 1);
  print_field("  ldown"  , 0x00c00020, 5, 1);
  print_word("MsgDelay"  , 0x00c00024);
  print_word("TxRefClks" , 0x00c00028);
  print_word("BuffByCnts", 0x00c0002c);
}

