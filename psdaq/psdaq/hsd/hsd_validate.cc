#include <chrono>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <pthread.h>
#include <poll.h>
#include "psdaq/hsd/Validator.hh"
#include "DataDriver.h"
#include "xtcdata/xtc/Dgram.hh"

static FILE* f = 0;
static unsigned _seconds=0;
static unsigned _nrxflags=0;
static bool     _lverbose = false;

static void sigHandler( int signal ) {
  psignal( signal, "Signal received by pgpWidget");
  if (f) fclose(f);

  printf("rxflags               : %u\n", _nrxflags);
  Validator::dump_totals();

  printf("Signal handler pulling the plug\n");
  ::exit(signal);
}


static void* diagnostics(void*)
{
  while(1) {
    sleep(1);
    Validator::dump_rates();
  }
  return 0;
}

#define EVENT_COUNT_ERR   0x01

static void show_usage(const char* p)
{
  printf("Usage: %s [options]\n",p);
  printf("Options: -d <device>\n");
  printf("         -f <output file>\n");
  printf("         -s <nskip> (analyze 1, skip n, ..)\n");
  printf("         -w <wait us>\n");
  printf("         -v (verbose)\n");
  printf("         -V <validate mask>\n");
  printf("            (bit 0 : event counter incr by 1)\n");
}

int main(int argc, char* argv[])
{

  int c;
  const char* pgpcard = "/dev/datadev_0";
  unsigned    lanem   = 0xf;

  const char* ofile = 0;
  int wait_us = 0;

  unsigned raw_start=  4, raw_rows = 20;
  unsigned fex_start=  4, fex_rows = 20;
  unsigned fex_thrlo=508, fex_thrhi=516;
  unsigned nskip = 0;
  bool     l134  = true;

  while((c = getopt(argc, argv, "d:f:s:w:vQ")) != EOF) {
    switch(c) {
    case 'd':
      pgpcard = optarg;
      break;
    case 'f':
      ofile = optarg;
      break;
    case 's':
      nskip = std::stoi(optarg, nullptr, 0);
      break;
    case 'w':
      wait_us =  std::stoi(optarg, nullptr, 16);
      break;
    case 'v':
      Validator::set_verbose(_lverbose = true);
      break;
    case 'Q':
      l134 = false;
      break;
    default:
      show_usage(argv[0]);
      return 0;
    }
  }

  Configuration cfg(raw_start, raw_rows,
                    fex_start, fex_rows,
                    fex_thrlo, fex_thrhi);

  ::signal( SIGINT, sigHandler );
  ::signal( SIGSEGV, sigHandler );

  //
  //  Launch the statistics thread
  //
  pthread_attr_t tattr;
  pthread_attr_init(&tattr);
  pthread_t thr;
  if (pthread_create(&thr, &tattr, &diagnostics, 0)) {
    perror("Error creating stat thread");
    return -1;
  }

  if (ofile) {
    f = fopen(ofile,"w");
    if (!f) {
      perror("Opening output file");
      exit(1);
    }
  }

  char err[256];
  int fd = open( pgpcard,  O_RDWR );
  if (fd < 0) {
    sprintf(err, "%s opening %s failed", argv[0], pgpcard);
    perror(err);
    return 1;
  }

  uint8_t mask[DMA_MASK_SIZE];
  dmaInitMaskBytes(mask);
  for(unsigned i=0; i<8; i++)
    if (lanem & (1<<i))
      dmaAddMaskBytes(mask, dmaDest(i,0));

  if (ioctl(fd, DMA_Set_MaskBytes, mask)<0) {
    perror("DMA_Set_MaskBytes");
    return -1;
  }

  Validator& val = l134 ? 
    *static_cast<Validator*>(new Fmc134Validator(cfg,5)) :
    *static_cast<Validator*>(new Fmc126Validator(cfg,0));

  const unsigned MAX_CNT = 128;
  unsigned getCnt = MAX_CNT;
  int32_t  dmaRet  [MAX_CNT];
  uint32_t dmaIndex[MAX_CNT];
  uint32_t rxFlags [MAX_CNT];
  unsigned dmaCount, dmaSize;
  void**   dmaBuffers;
  if ( (dmaBuffers = dmaMapDma(fd,&dmaCount,&dmaSize)) == NULL ) {
    printf("Failed to map dma buffers\n");
    return -1;
  }

  unsigned iskip = nskip;

  // DMA Read
  do {
    pollfd pfd;
    pfd.fd      = fd;
    pfd.events  = POLLIN;
    pfd.revents = 0;

    int result = poll(&pfd, 1, -1);
    if (result < 0) {
      perror("poll");
      return -1;
    }

    int bret = dmaReadBulkIndex(fd,getCnt,dmaRet,dmaIndex,rxFlags,NULL,NULL);

    for(int idg=0; idg<bret; idg++) {

      int ret = dmaRet[idg];

      if (rxFlags[idg])
        _nrxflags++;

      char* data = (char*)dmaBuffers[dmaIndex[idg]];

      if (f)
        fwrite(data, ret, 1, f);

      XtcData::Transition* event_header = reinterpret_cast<XtcData::Transition*>(data);
      XtcData::TransitionId::Value transition_id = event_header->seq.service();

      if (_lverbose) {
        const uint32_t* pu32 = reinterpret_cast<const uint32_t*>(data);
        const uint64_t* pu64 = reinterpret_cast<const uint64_t*>(data);
        printf("Data: %016lx %016lx %08x %08x %08x %08x\n", pu64[0], pu64[1], pu32[4], pu32[5], pu32[6], pu32[7]);
        printf("Size %u B | Flags %x | Transition id %d | pulse id %lx | event counter %x\n",
               ret, rxFlags[idg], transition_id, event_header->seq.pulseId().value(), *reinterpret_cast<uint32_t*>(event_header+1));
      }

      if (ret>0)
        if (!iskip--) {
          val.validate(data,ret);
          iskip = nskip;
        }
      if (wait_us && event_header->seq.stamp().seconds()>_seconds) {
        usleep(wait_us);
        _seconds = event_header->seq.stamp().seconds();
      }
    }

    if (bret>0)
      dmaRetIndexes(fd,bret,dmaIndex);

  } while (1);

  pthread_join(thr,NULL);

  return 0;
}
