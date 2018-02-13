
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>

#include <string>
#include <vector>

extern int optind;

#define MOD_SHARED 12
#define MAX_TPR_ALLQ (32*1024)
#define MAX_TPR_CHNQ  1024
#define MSG_SIZE      32

namespace Pds {
  namespace Tpr {

// DMA Buffer Size, Bytes (could be as small as 512B)
#define BUF_SIZE 4096
#define NUMBER_OF_RX_BUFFERS 256

    class TprEntry {
    public:
      uint32_t word[MSG_SIZE];
    };

    class ChnQueue {
    public:
      TprEntry entry[MAX_TPR_CHNQ];
    };

    class TprQIndex {
    public:
      long long idx[MAX_TPR_ALLQ];
    };

    class TprQueues {
    public:
      TprEntry  allq  [MAX_TPR_ALLQ];
      ChnQueue  chnq  [MOD_SHARED];
      TprQIndex allrp [MOD_SHARED]; // indices into allq
      long long        allwp [MOD_SHARED]; // write pointer into allrp
      long long        chnwp [MOD_SHARED]; // write pointer into chnq's
      long long        gwp;
      int              fifofull;
    };
  };
};

using namespace Pds::Tpr;

static void dumpFrame(const uint32_t* p)
{
  const uint64_t* pl = reinterpret_cast<const uint64_t*>(p);
  char m = p[0]&(1<<30) ? 'D':' ';
  //
  //  We only expect BSA_CHN messages in this queue
  //
  switch(p[1]&0xffff) {
  case 0:
    printf("EVENT [x%x]: %16lx %16lx %16lx %16lx %16lx %c\n",
           (p[1]>>16)&0xffff,pl[0],pl[1],pl[2],pl[3],pl[4],m);
    break;
  case 1:
    printf("BSA_CNTL : %16lx %16lx %16lx %16lx %c\n",
           pl[0],pl[1],pl[2],pl[3],m);
    break;
  case 2:
    printf("BSA_CHN [x%x]: %16lx %16lx %16lx %16lx %c\n",
           (p[1]>>16)&0xff,pl[0],pl[1],pl[2],pl[3],m);
    break;
  default:
    break;
  }
}

static uint64_t eventFrames     =0;
static uint64_t bsaControlFrames=0;
static uint64_t bsaChannelFrames=0;

static void countFrame(const uint32_t* p)
{
  //  const uint64_t* pl = reinterpret_cast<const uint64_t*>(p);
  //  char m = p[0]&(1<<30) ? 'D':' ';
  //
  //  We only expect BSA_CHN messages in this queue
  //
  switch(p[1]&0xffff) {
  case 0:
    eventFrames++;
    break;
  case 1:
    bsaControlFrames++;
    break;
  case 2:
    bsaChannelFrames++;
    break;
  default:
    break;
  }
}
static void* read_thread(void*);

void usage(const char* p) {
  printf("Usage: %s -r <a/b> -i <channel> [-v]\n",p);
}

int main(int argc, char** argv) {

  extern char* optarg;
  char tprid='a';
  int idx=0;
  bool lverbose=false;

  int c;
  bool lUsage = false;
  while ( (c=getopt( argc, argv, "r:i:vh?")) != EOF ) {
    switch(c) {
    case 'r':
      tprid  = optarg[0];
      if (strlen(optarg) != 1) {
        printf("%s: option `-r' parsing error\n", argv[0]);
        lUsage = true;
      }
      break;
    case 'i':
      idx = atoi(optarg);
      break;
    case 'v':
      lverbose = true;
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
    char dev[16];
    sprintf(dev,"/dev/tpr%c%c",tprid,idx<0 ? 'b': (idx<10 ? '0'+idx : 'a'+(idx-10)));
    printf("Using tpr %s\n",dev);

    int fd = open(dev, O_RDWR);
    if (fd<0) {
      perror("Could not open");
      return -1;
    }

    void* ptr = mmap(0, sizeof(TprQueues), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
      perror("Failed to map");
      return -2;
    }

    if (!lverbose) {
      pthread_attr_t tattr;
      pthread_attr_init(&tattr);
      pthread_t tid;
      if (pthread_create(&tid, &tattr, &read_thread, 0))
        perror("Error creating read thread");
    }

    TprQueues& q = *(TprQueues*)ptr;

    char* buff = new char[32];

    if (idx>=0) {
      int64_t allrp = q.allwp[idx];
      int64_t chnrp = q.chnwp[idx];
      while(1) {
        read(fd, buff, 32);
        if (lverbose) { 
          printf("allwp 0x%llx,  chnwp 0x%llx,  gwp 0x%llx\n",
                 q.allwp[idx],
                 q.chnwp[idx],
                 q.gwp);
          while(chnrp < q.chnwp[idx]) {
            dumpFrame(reinterpret_cast<const uint32_t*>(&q.chnq[idx].entry[chnrp &(MAX_TPR_CHNQ-1)].word[0]));
            chnrp++;
          }
          while(allrp < q.allwp[idx]) {
            dumpFrame(reinterpret_cast<const uint32_t*>
                      (&q.allq[q.allrp[idx].idx[allrp &(MAX_TPR_ALLQ-1)] &(MAX_TPR_ALLQ-1) ].word[0]));
            allrp++;
          }
        }
        else {
          while(chnrp < q.chnwp[idx]) {
            countFrame(reinterpret_cast<const uint32_t*>(&q.chnq[idx].entry[chnrp &(MAX_TPR_CHNQ-1)].word[0]));
            chnrp++;
          }
          while(allrp < q.allwp[idx]) {
            countFrame(reinterpret_cast<const uint32_t*>
                      (&q.allq[q.allrp[idx].idx[allrp &(MAX_TPR_ALLQ-1)] &(MAX_TPR_ALLQ-1) ].word[0]));
            allrp++;
          }
        }
      }
    }
    else {
      int64_t grp = q.gwp;
      while(1) {
        while(grp == q.gwp)
          sleep(1);
        printf("allwp 0x%llx,  chnwp 0x%llx,  gwp 0x%llx\n",
               q.allwp[idx],
               q.chnwp[idx],
               q.gwp);
        while(grp < q.gwp) {
          dumpFrame(reinterpret_cast<const uint32_t*>
                    (&q.allq[grp &(MAX_TPR_ALLQ-1) ].word[0]));
          grp++;
        }
      }
    }
  }

  return 0;
}

void* read_thread(void* arg)
{
  uint64_t evFrames=eventFrames;
  uint64_t ctlFrames=bsaControlFrames;
  uint64_t chnFrames=bsaChannelFrames;

  while(1) {
    sleep(1);
    printf("EvFrames %9llu : BsaCntl %9llu : BsaChan %9llu\n",
           (unsigned long long)(eventFrames-evFrames),
           (unsigned long long)(bsaControlFrames-ctlFrames),
           (unsigned long long)(bsaChannelFrames-chnFrames));
    evFrames=eventFrames;
    ctlFrames=bsaControlFrames;
    chnFrames=bsaChannelFrames;
  }

  return 0;
}
