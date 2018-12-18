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

#define MAX_RET_CNT_C 1000

#define EVENT_COUNT_ERR   0x01
#define FRAME_COUNT_ERR   0x02
#define FRAME_CONTENT_ERR 0x04
#define FRAME_SIZE_ERR    0x08

#define HISTORY_ERR       0x10
#define LANE_MISS_ERR     0x20
#define PULSE_ID_ERR      0x40
#define FRAME_COUNT_MISM  0x80

static unsigned verbose = 0;
static FILE* fdump = 0;

static void usage(const char* p) {
  printf("Usage: %s [options'\n",p);
  printf("Options:\n");
  printf("\t-d <device file>     (default: /dev/datadev_1)\n");
  printf("\t-f <dump filename>   (default: none)\n");
  printf("\t-c <update interval> (default: 1000000)\n");
  printf("\t-v <test mask>       (default: 0)\n");
}

#define HISTORY 256
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
  { memset(_lanes,0,HISTORY*sizeof(unsigned)); }
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
    unsigned event = p[4]&0xffffff;
    unsigned now = event % HISTORY;
    unsigned pre = (event-1) % HISTORY;
    _ncalls++;

    if (_lanes[now] & (1<<lane)) {
      _historyErr++;
      if ((verbose & HISTORY_ERR) && nprint++ < MAX_PRINT)
        printf("\thistory : %x [%x]\n", _lanes[now], 1<<lane);
    }
    
    if ((_lanes[now] |= (1<<lane))==_lanemask) {
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
    nprint = 0;
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
  unsigned _lanes  [HISTORY];
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
EventValidator eventv(lanev,0xf);

void LaneValidator::validate(const uint32_t* p, unsigned sz) {
  _ncalls++;
  unsigned event = p[4]&0xffffff;

  //  validate this event
  unsigned now = event%HISTORY;
  _pulseId [now] = *reinterpret_cast<const uint64_t*>(p);
  _frameCnt[now] = p[8];
  for(unsigned i=1; i<sz-9; i++)
    if (p[sz-i]!=i) {
      _frameContentErr++;
      if (verbose & FRAME_CONTENT_ERR)
        printf("\t[%p] [%06x.%ld]: p[%d] %x\n", p, event, this-lanev, i, p[sz-i]);
      if (fdump) {
        fwrite(&sz,sizeof(unsigned),1,fdump);
        fwrite(p  ,sizeof(unsigned),sz,fdump);
      }
      break;
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
      if (verbose & EVENT_COUNT_ERR)
        printf("\t[%p] [%06x.%ld] eventCount %06x [%06x]\n",
               p, event, this-lanev, event,_current);
      _current = event;
    }
    if (p[8]!=_frameCnt[pre]+1) {
      _frameCounterErr++;
      if (verbose & FRAME_COUNT_ERR) {
        printf("\t[%p] [%06x.%ld] frame %08x [%08x]\n",
               p, event, this-lanev, p[8],_frameCnt[pre]+1);
      }
    }
    if (_sz != sz) {
      _frameSizeErr++;
      if (verbose & FRAME_SIZE_ERR)
        printf("\t[%p] [%06x.%ld] frameSize : %x [%x]\n", 
               p, event, this-lanev, sz, _sz);
    }
  }
}

void sigHandler( int signal ) {
  psignal( signal, "Signal received by pgpWidget");
  eventv.summary();
  for(unsigned i=0; i<4; i++)
    lanev[i].summary();
  ::exit(signal);
}


int main (int argc, char **argv) {
   uint8_t       mask[DMA_MASK_SIZE];
   int32_t       ret;
   int32_t       s;
   uint32_t      rxFlags[MAX_RET_CNT_C];
   uint32_t      dest   [MAX_RET_CNT_C];
   void **       dmaBuffers;
   uint32_t      dmaSize;
   uint32_t      dmaCount;
   uint32_t      dmaIndex[MAX_RET_CNT_C];
   int32_t       dmaRet[MAX_RET_CNT_C];
   int32_t       x;
   float         last;
   float         rate;
   float         bw;
   float         duration;
   int32_t       max;
   int32_t       total;

   uint32_t      getCnt = MAX_RET_CNT_C;

   struct timeval sTime;
   struct timeval eTime;
   struct timeval dTime;
   struct timeval pTime[7];

   unsigned count = 1000000;
   const char* dev  = 0;
   const char* dump = 0;
   extern char* optarg;
   int c;
   while((c=getopt(argc,argv,"d:f:c:v:"))!=EOF) {
     switch(c) {
     case 'd': dev     = optarg; break;
     case 'f': dump    = optarg; break;
     case 'c': count   = strtoul(optarg,NULL,0); break;
     case 'v': verbose = strtoul(optarg,NULL,0); break;
     default: usage(argv[0]); return 1;
     }
   }

   ::signal( SIGINT, sigHandler );

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
      printf("Failed to map dma buffers!\n");
      return(0);
   }
#endif

   dmaSetMaskBytes(s,mask);

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
               if (b[5]>>31) {  // L1Accept
                 lanev[lane].validate(b,words);
                 eventv     .validate(b,words,lane);
               }
               else {           // Other
                 printf("lane%u:",lane);
                 for(unsigned i=0; i<words; i++)
                   printf(" %08x",b[i]);
                 printf("\n");
               }
               //  Print out pulseId/timeStamp to check synchronization across nodes
               if ((b[4]&0xfffff)==0) {
                 printf("event[%06x]:",b[4]);
                 for(unsigned i=0; i<6; i++)
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
      eventv.report();
      for(unsigned i=0; i<4; i++)
        lanev[i].report();

      rate = 0.0;
      bw   = 0.0;
   }

   return(0);
}

