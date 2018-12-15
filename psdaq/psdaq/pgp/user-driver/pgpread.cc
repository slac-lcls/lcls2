#include <chrono>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <pthread.h>
#include "pgpdriver.h"
#include "xtcdata/xtc/Dgram.hh"

static unsigned _nevents=0;
static uint64_t _nbytes =0;
static unsigned _lanes  =0;

static void printrate(const char* name,
                      const char* units,
                      double      rate)
{
  static const char scchar[] = { ' ', 'k', 'M', 'G', 'T' };
  unsigned rsc = 0;

  if (rate > 1.e12) {
    rsc     = 4;
    rate   *= 1.e-12;
  }
  else if (rate > 1.e9) {
    rsc     = 3;
    rate   *= 1.e-9;
  }
  else if (rate > 1.e6) {
    rsc     = 2;
    rate   *= 1.e-6;
  }
  else if (rate > 1.e3) {
    rsc     = 1;
    rate   *= 1.e-3;
  }

  printf("%s %7.2f %c%s", name, rate, scchar[rsc], units);
}

static void* diagnostics(void*)
{
  timespec tv;
  clock_gettime(CLOCK_REALTIME,&tv);
  unsigned nevents=_nevents;
  uint64_t nbytes =_nbytes;
  _lanes = 0;
  while(1) {
    sleep(1);
    timespec ttv;
    clock_gettime(CLOCK_REALTIME,&ttv);
    unsigned nev = _nevents;
    uint64_t nby = _nbytes;
    unsigned lanes = _lanes;
    _lanes = 0;
    double dt = double(ttv.tv_sec - tv.tv_sec) + 1.e-9*(double(ttv.tv_nsec)-double(tv.tv_nsec));
    double revents = double(nev-nevents)/dt;
    double rbytes  = double(nby -nbytes)/dt;
    printrate("\t ", "Hz", revents);
    printrate("\t ", "B/s", rbytes);
    printf("\t lanes %x\n", lanes);
    nevents=nev;
    nbytes =nby;
    tv     =ttv;
  }
  return 0;
}

int main(int argc, char* argv[])
{
    
    int c;
    int device_id;
    const char* ofile = 0;
    int wait_us = 0;
    bool lverbose = false;
    while((c = getopt(argc, argv, "d:f:w:v")) != EOF) {
        switch(c) {
        case 'd':
          device_id = std::stoi(optarg, nullptr, 16);
          break;
        case 'f':
          ofile = optarg;
          break;
        case 'w':
          wait_us =  std::stoi(optarg, nullptr, 16);
          break;
        case 'v':
          lverbose = true;
          break;
        }
    }


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

    FILE* f = 0;
    if (ofile) {
      f = fopen(ofile,"w");
      if (!f) {
        perror("Opening output file");
        exit(1);
      }
    }

    int num_entries = 8192;
    DmaBufferPool pool(num_entries, RX_BUFFER_SIZE);
    AxisG2Device dev(device_id);
    dev.init(&pool);       
    dev.setup_lanes(0xF);
    unsigned _seconds = 0;

    while (true) {    
        DmaBuffer* buffer = dev.read();
        XtcData::Transition* event_header = reinterpret_cast<XtcData::Transition*>(buffer->virt);
        XtcData::TransitionId::Value transition_id = event_header->seq.service();
        _nevents++;
        _nbytes += buffer->size;
        _lanes  |= 1<<buffer->dest;
        if (lverbose) {
          printf("Size %u B | Dest %u | Transition id %d | pulse id %lu | event counter %u\n",
                 buffer->size, buffer->dest, transition_id, event_header->seq.pulseId().value(), event_header->evtCounter); 
        }
        if (wait_us && event_header->seq.stamp().seconds()>_seconds) {
          usleep(wait_us);
          _seconds = event_header->seq.stamp().seconds();
        }
        if (f)
          fwrite(buffer->virt, buffer->size, 1, f);

        dev.read_done(buffer);
    }                                

  pthread_join(thr,NULL);
  return 0;
} 
