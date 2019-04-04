#include <unistd.h>
#include <stdio.h>

#include "xtcdata/xtc/Dgram.hh"
#include "psdaq/hsd/Validator.hh"

static bool     _lverbose = false;
static unsigned _nevents=0;
static uint64_t _nbytes =0;
static uint64_t _rbytes =0;
static unsigned _revents=0;
static uint64_t _fbytes =0;
static unsigned _fevents=0;
static unsigned _overrun=0;
static unsigned _corrupt=0;

static uint64_t _initial_sample_errors=0;
static uint64_t _internal_sample_errors=0;
static uint64_t _finitial_sample_errors=0;
static uint64_t _finternal_sample_errors=0;
static unsigned _nprints=20;

static FILE* fdump = 0;

void Validator::set_verbose(bool v) { _lverbose=v; }

void Validator::dump_totals() {
  if (fdump) fclose(fdump);

  printf("==========\n");
  printf("events                : %u\n", _nevents);
  printf("bytes                 : %lu\n", _nbytes);
  printf("--\n");
  printf("raw events            : %u\n", _revents);
  printf("raw bytes             : %lu\n", _rbytes);
  printf("initial sample errors : %lu\n", _initial_sample_errors);
  printf("internal sample errors: %lu\n", _internal_sample_errors);
  printf("fex events            : %u\n", _fevents);
  printf("fex bytes             : %lu\n", _fbytes);
  printf("finitial sample error : %lu\n", _finitial_sample_errors);
  printf("finternal sample error: %lu\n", _finternal_sample_errors);
  printf("overrun               : %u\n", _overrun);
  printf("corrupt               : %u\n", _corrupt);
  printf("==========\n");
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

void Validator::dump_rates()
{
  static timespec tv;
  static unsigned nevents=0;
  static uint64_t nbytes ;
  static unsigned revents;
  static uint64_t rbytes ;
  static unsigned fevents;
  static uint64_t fbytes ;

  timespec ttv;
  clock_gettime(CLOCK_REALTIME,&ttv);

  if (_nevents) {
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
  }
  else {
    nevents=_nevents;
    nbytes =_nbytes;
    revents=_revents;
    rbytes =_rbytes;
    fevents=_fevents;
    fbytes =_fbytes;
  }
  tv     =ttv;
}

static unsigned ndump=10;
//static unsigned ndump=0;
static void dump(const char* b, int nb)
{
  if (ndump) {
    if (!fdump) 
      fdump = fopen("/tmp/hsd_validate.dump","w");
    fwrite(b,1,nb,fdump);
    ndump--;
  }
}

Validator::Validator(const Configuration& cfg) : 
    _cfg         (cfg)
  { _transition.env = 0; }

void Validator::validate(const char* buffer, int ret)
{
  const XtcData::Transition* event_header = reinterpret_cast<const XtcData::Transition*>(buffer);
  _nevents++;
  _nbytes += ret;

  if (event_header->seq.isEvent()) {
    unsigned streams(buffer[26]>>4);
    const char* p = buffer+32;
    //      bool ldump=false;
    bool ldump=true;
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
          //            p -= 32; // These errors drop 32B, correct for the next stream
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

bool Validator::_validate_raw(const XtcData::Transition& transition,
                              const StreamHeader&        s) {
  bool lret=false;
  // Decode stream header
  _revents++;
  _rbytes += s.num_samples()*2;

  if (_lverbose) {
    printf("raw header %04x:%04x %04x[%04x]\n",
           s._p[3]&0xffff,s._p[3]>>16,s.num_samples(),s.cache_len());
  }

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

bool Validator::_validate_fex(const StreamHeader&        s) {
  bool lret=false;
  // Decode stream header
  _fevents++;
  _fbytes += s.num_samples()*2;

  if (_lverbose) {
    printf("fex header %04x:%04x %04x[%04x]\n",
           s._p[3]&0xffff,s._p[3]>>16,s.num_samples(),s.cache_len());
  }

  if (1) {
    //  calculate expected sample value
    uint16_t sample_value = _sample_value;

    const uint16_t* q = reinterpret_cast<const uint16_t*>(&s+1);

    unsigned k=0;
    unsigned ns=0;
    while(q[k]&0x8000) {  // skipped samples
      ns += (q[k]&0x7fff)+1;
      k++;
    }
    sample_value += ns>>2;

    if (k>=s.num_samples())
      return lret;

    if ((sample_value&0x7ff)!=q[k]) {  // initial sample error
      _finitial_sample_errors++;
      lret = true;
      if (_lverbose || _nprints) {
        _nprints--;
        printf("Expected initial fex value %04x [%04x]\n", sample_value&0x7ff, q[k]);
      }
    }

    sample_value = q[k];

    bool lerr=false;
    for(unsigned i=k; i<s.num_samples() && !lerr; ) {
      if (q[i]&0x8000) {
        unsigned ns=0; 
        for(unsigned j=0; j<4; j++,i++)
          ns += (q[i]&0x7fff);
        sample_value += ns>>2;
      }
      else {
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
    }

    if (lerr) {
      _finternal_sample_errors++;
      lret = true;
    }
  }

  return lret;
}

