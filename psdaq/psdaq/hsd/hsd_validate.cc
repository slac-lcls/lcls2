#include <chrono>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <pthread.h>
#include <poll.h>
#include "psdaq/pgp/kcu1500/app/DataDriver.h"

#include "xtcdata/xtc/Dgram.hh"

static bool     _lverbose = false;
static unsigned _nevents=0;
static uint64_t _nbytes =0;
static unsigned _seconds=0;
static uint64_t _rbytes =0;
static unsigned _revents=0;
static uint64_t _fbytes =0;
static unsigned _fevents=0;
static unsigned _nrxflags=0;
static unsigned _overrun=0;
static unsigned _corrupt=0;

static uint64_t _initial_sample_errors=0;
static uint64_t _internal_sample_errors=0;
static uint64_t _finitial_sample_errors=0;
static uint64_t _finternal_sample_errors=0;
static unsigned _nprints=20;

static FILE* f = 0;
static FILE* fdump = 0;

static void sigHandler( int signal ) {
  psignal( signal, "Signal received by pgpWidget");
  if (f) fclose(f);
  if (fdump) fclose(fdump);

  printf("==========\n");
  printf("events                : %u\n", _nevents);
  printf("bytes                 : %lu\n", _nbytes);
  printf("rxflags               : %u\n", _nrxflags);
  printf("--\n");
  printf("raw events            : %u\n", _revents);
  printf("raw bytes             : %lu\n", _rbytes);
  printf("initial sample errors : %lu\n", _initial_sample_errors);
  printf("internal sample errors: %lu\n", _internal_sample_errors);
  printf("finitial sample error : %lu\n", _finitial_sample_errors);
  printf("finternal sample error: %lu\n", _finternal_sample_errors);
  printf("overrun               : %u\n", _overrun);
  printf("corrupt               : %u\n", _corrupt);
  printf("==========\n");

  printf("Signal handler pulling the plug\n");
  ::exit(signal);
}


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
  unsigned revents=_revents;
  uint64_t rbytes =_rbytes;
  unsigned fevents=_fevents;
  uint64_t fbytes =_fbytes;
  while(1) {
    sleep(1);
    timespec ttv;
    clock_gettime(CLOCK_REALTIME,&ttv);
    unsigned nev = _nevents;
    uint64_t nby = _nbytes;
    unsigned rev = _revents;
    uint64_t rby = _rbytes;
    unsigned fev = _fevents;
    uint64_t fby = _fbytes;
    double dt = double(ttv.tv_sec - tv.tv_sec) + 1.e-9*(double(ttv.tv_nsec)-double(tv.tv_nsec));
    double devents = double(nev-nevents)/dt;
    double dbytes  = double(nby -nbytes)/dt;
    unsigned bev   = nev==nevents ? 0 : (nby-nbytes)/(nev-nevents);
    unsigned rbev  = rev==revents ? 0 : (rby-rbytes)/(rev-revents);
    unsigned fbev  = fev==fevents ? 0 : (fby-fbytes)/(fev-fevents);
    printrate("\t ", "Hz", devents);
    printrate("\t ", "B/s", dbytes);
    printf("\t %u B/ev", bev);
    printf("\t %u B/rev", rbev);
    printf("\t %u B/fev\n", fbev);
    nevents=nev;
    nbytes =nby;
    revents=rev;
    rbytes =rby;
    fevents=fev;
    fbytes =fby;
    tv     =ttv;
  }
  return 0;
}

static unsigned ndump=10;
static void dump(const char* b, int nb)
{
  if (ndump) {
    if (!fdump) 
      fdump = fopen("/tmp/hsd_validate.dump","w");
    fwrite(b,1,nb,fdump);
    ndump--;
  }
}

class Configuration {
public:
public:
  Configuration(unsigned raw_gate_start,
                unsigned raw_gate_rows,
                unsigned fex_gate_start,
                unsigned fex_gate_rows,
                unsigned fex_thresh_lo,
                unsigned fex_thresh_hi) :
    _raw_gate_start (raw_gate_start),
    _raw_gate_rows  (raw_gate_rows ),
    _fex_gate_start (fex_gate_start),
    _fex_gate_rows  (fex_gate_rows ),
    _fex_thresh_lo  (fex_thresh_lo ),
    _fex_thresh_hi  (fex_thresh_hi )
  {}
public:
  unsigned _raw_gate_start;
  unsigned _raw_gate_rows;
  unsigned _fex_gate_start;
  unsigned _fex_gate_rows;
  unsigned _fex_thresh_lo;
  unsigned _fex_thresh_hi;
};

class StreamHeader {
public:
  StreamHeader() {}
public:
  unsigned num_samples() const { return _p[0]&~(1<<31); }
  unsigned stream_id  () const { return (_p[1]>>24)&0xff; }
private:
  uint32_t _p[4];
};

class Validator {
public:
  Validator(const Configuration& cfg) : 
    _cfg         (cfg)
  { _transition.env = 0; }
public:
  void validate(const char* buffer, int ret)
  {
    const XtcData::Transition* event_header = reinterpret_cast<const XtcData::Transition*>(buffer);
    _nevents++;
    _nbytes += ret;

    if (event_header->seq.isEvent()) {
      unsigned streams(buffer[26]>>4);
      const char* p = buffer+32;
      bool ldump=false;
      while(streams) {
        const StreamHeader& s = *reinterpret_cast<const StreamHeader*>(p);
        if (p >= buffer+ret) {
          _overrun++;
          if (!ldump)
            dump(buffer,ret);
          return;
        }
        if ((streams & (1<<s.stream_id()))==0) {
          _corrupt++;
          if (!ldump)
            dump(buffer,ret);
          return;
        }
        if (s.stream_id() == 0)
          if (_validate_raw(*event_header,s)) {
            ldump=true;
            dump(buffer,ret);
            p -= 32; // These errors drop 32B, correct for the next stream
          }
        if (s.stream_id() == 1)
          if (_validate_fex(s))
            if (!ldump) {
              ldump=true;
              dump(buffer,ret);
            }
        p += sizeof(StreamHeader)+s.num_samples()*2;
        streams &= ~(1<<s.stream_id());
      }
    }
  }
private:
  bool _validate_raw(const XtcData::Transition& transition,
                     const StreamHeader&        s) {
    bool lret=false;
    // Decode stream header
    _revents++;
    _rbytes += s.num_samples()*2;

    if (_transition.env) {
      //  calculate expected sample value
      uint16_t sample_value = _sample_value;
      sample_value += ( transition.seq.pulseId().value() - 
                        _transition.seq.pulseId().value()) * 1348;

      const uint16_t* q = reinterpret_cast<const uint16_t*>(&s+1);

      if ((sample_value&0x7ff)!=q[0]) {  // initial sample error
        _initial_sample_errors++;
        lret=true;
        if (_lverbose || _nprints) {
          _nprints--;
          printf("Expected initial raw value %04x [%04x]\n", sample_value&0x7ff, q[0]);
        }
      }

      sample_value = q[0];

      bool lerr=false;
      for(unsigned i=0; i<s.num_samples(); ) {
        bool verr=false;
        unsigned j;
        for(j=0; j<4 && !verr; j++,i++)
          if ((sample_value&0x7ff) != q[i]) { // internal sample error
            if ((_lverbose || _nprints) && !lerr) {
              _nprints--;
              printf("Expected raw value %04x [%u] :",sample_value&0x7ff, i);
              for(unsigned k=i>3 ? i-4:0; k<i+4; k++)
                printf(" %04x",q[k]);
              printf("\n");
            }
            verr=true;
          }
        if (verr) {
          i += 4-j;
          lerr = true;
        }
        sample_value++;
      }

      if (lerr) {
        _internal_sample_errors++;
        lret = true;
      }
    }

    _transition = transition;
    _sample_value = *reinterpret_cast<const uint16_t*>(&s+1);
    return lret;
  }
  bool _validate_fex(const StreamHeader&        s) {
    bool lret=false;
    // Decode stream header
    _fevents++;
    _fbytes += s.num_samples()*2;

    if (1) {
      //  calculate expected sample value
      uint16_t sample_value = _sample_value;

      const uint16_t* q = reinterpret_cast<const uint16_t*>(&s+1);

      if ((sample_value&0x7ff)!=q[0]) {  // initial sample error
        _finitial_sample_errors++;
        lret = true;
        if (_lverbose || _nprints) {
          _nprints--;
          printf("Expected initial fex value %04x [%04x]\n", sample_value&0x7ff, q[0]);
        }
      }

      sample_value = q[0];

      bool lerr=false;
      for(unsigned i=0; i<s.num_samples() && !lerr; ) {
        for(unsigned j=0; j<4 && !lerr; j++,i++)
          if ((sample_value&0x7ff) != q[i]) { // internal sample error
            if ((_lverbose || _nprints) && !lerr) {
              _nprints--;
              printf("Expected fex value %04x [%u] :",sample_value&0x7ff, i);
              for(unsigned k=i>3 ? i-4:0; k<i+4; k++)
                printf(" %04x",q[k]);
              printf("\n");
            }
            lerr=true;
          }
        sample_value++;
      }

      if (lerr) {
        _finternal_sample_errors++;
        lret = true;
      }
    }

    return lret;
  }
private:
  const Configuration& _cfg;
  XtcData::Transition  _transition;
  unsigned             _sample_value;
};

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

  while((c = getopt(argc, argv, "d:f:s:w:v")) != EOF) {
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
      _lverbose = true;
      break;
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

  Validator val(cfg);

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
      if (f)
        fwrite(data, ret, 1, f);
    }

    if (bret>0)
      dmaRetIndexes(fd,bret,dmaIndex);

  } while (1);
  
  pthread_join(thr,NULL);

  return 0;
} 
