
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

#include "psdaq/tpr/Queues.hh"

extern int optind;

namespace Pds {
  namespace Tpr {
    class TA {
    public:
      TA(void* _ptr, int _fd) : ptr(_ptr), fd(_fd) {}
    public:
      void* ptr;
      int   fd;
    };
  }
}

using namespace Pds::Tpr;

static bool lverbose = false;

static void dumpFrame(const uint32_t* p)
{
  if (lverbose) {
    printf("dumpFrame:");
    for(unsigned i=0; i<8; i++)
      printf(" %08x", p[i]);
    printf("\n");
  }

  const uint64_t* pl;
  char m = p[0]&(1<<30) ? 'D':' ';
  unsigned mtyp = (p[0]>>16)&0x3;
  //
  //  We only expect BSA_CHN messages in this queue
  //
  switch(mtyp) {
  case 0:
    pl = reinterpret_cast<const uint64_t*>(p+2);
    printf("EVENT [x%x]: %16lx %16lx %16lx %16lx %16lx %c\n",
           (p[1]>>16)&0xffff,pl[0],pl[1],pl[2],pl[3],pl[4],m);
    break;
  case 1:
    pl = reinterpret_cast<const uint64_t*>(p+1);
    printf("BSA_CNTL : %u.%09u %16lx I%16lx m%16lx M%16lx %c\n",
           p[4],p[3],pl[0],pl[2],pl[3],pl[4],m);
    break;
  case 2:
    pl = reinterpret_cast<const uint64_t*>(p+1);
    printf("BSA_CHN [x%x]: %u.%09u %16lx A%16lx D%16lx U%16lx %c\n",
           (p[1]>>16)&0xff,p[8],p[7],pl[0],pl[1],pl[2],pl[4],m);
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
  unsigned mtyp = (p[0]>>16)&0x3;
  switch(mtyp) {
  case 0:
    eventFrames++;
    break;
  case 1:
    bsaControlFrames++;
    dumpFrame(p);
    break;
  case 2:
    bsaChannelFrames++;
    break;
  default:
    break;
  }
}
static void* read_thread(void*);
static void* bsa_thread (void*);

void usage(const char* p) {
  printf("Usage: %s -r <a/b> -i <channel> [-v]\n",p);
}

int main(int argc, char** argv) {

  extern char* optarg;
  char tprid='a';
  int idx=-1;

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
    const char* sfx[] = {"0","1","2","3","4","5","6","7","8","9","a","b","c","d",NULL};
    char dev[16];
    sprintf(dev,"/dev/tpr%c%s",tprid,idx<0 ? "BSA":sfx[idx]);
    printf("Using tpr %s\n",dev);

    int fd = open(dev, O_RDWR);
    if (fd<0) {
      perror("Could not open");
      return -1;
    }

    sprintf(dev,"/dev/tpr%cBSA",tprid);
    int fd_bsa = open(dev, O_RDWR);
    if (fd_bsa<0) {
      perror("Could not open");
      return -1;
    }

    void* ptr = mmap(0, sizeof(Pds::Tpr::Queues), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
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

    if (!lverbose) {
      pthread_attr_t tattr;
      pthread_attr_init(&tattr);
      pthread_t tid;
      TA* arg = new TA(ptr, fd_bsa);
      if (pthread_create(&tid, &tattr, &bsa_thread, arg))
        perror("Error creating bsa thread");
    }

    Pds::Tpr::Queues& q = *(Pds::Tpr::Queues*)ptr;

    char* buff = new char[32];

    if (idx>=0) {
      int64_t rp = q.allwp[idx];
      if (lverbose) { 
        printf("allwp 0x%llx,  bsawp 0x%llx,  gwp 0x%llx\n",
               q.allwp[idx],
               q.bsawp,
               q.gwp);
      }
      while(1) {
        read(fd, buff, 32);
        if (lverbose) { 
          printf("allwp 0x%llx,  bsawp 0x%llx,  gwp 0x%llx\n",
                 q.allwp[idx],
                 q.bsawp,
                 q.gwp);
          while(rp < q.allwp[idx]) {
            long long qi = q.allrp[idx].idx[rp%MAX_TPR_ALLQ]%MAX_TPR_ALLQ;
            dumpFrame(reinterpret_cast<const uint32_t*>(&q.allq[qi].word[0]));
            rp++;
          }
        }
        else {
          while(rp < q.allwp[idx]) {
            long long qi = q.allrp[idx].idx[rp%MAX_TPR_ALLQ]%MAX_TPR_ALLQ;
            countFrame(reinterpret_cast<const uint32_t*>(&q.allq[qi].word[0]));
            rp++;
          }
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

void* bsa_thread(void* arg)
{
  TA* ta = (TA*)arg;
  Pds::Tpr::Queues& q = *(Pds::Tpr::Queues*)ta->ptr;
  int fd = ta->fd;
  char* buff = new char[32];

  int64_t rp = q.bsawp;
  while(1) {
    read(fd, buff, 32);
    while(rp < q.bsawp) {
      long long qi = rp%MAX_TPR_BSAQ;
      countFrame(reinterpret_cast<const uint32_t*>(&q.bsaq[qi].word[0]));
      rp++;
    }
  }
}
