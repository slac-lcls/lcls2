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
  static unsigned fill[] = { 0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef };
  if (ndump) {
    if (!fdump) 
      fdump = fopen("/tmp/hsd_validate.dump","w");
    fwrite(b,1,nb,fdump);
    fwrite(fill,4,sizeof(unsigned),fdump);
    ndump--;
  }
}

Validator::Validator(const Configuration& cfg) : 
  _cfg         (cfg),
  _neventcnt   (0)
  { _transition.env = 0; }

void Validator::validate(const char* buffer, int ret)
{
  bool ldump=false;
  //    bool ldump=true;

  const XtcData::Transition* event_header = reinterpret_cast<const XtcData::Transition*>(buffer);

  if (_nevents) {
    const uint32_t* xenv = reinterpret_cast<const uint32_t*>(event_header+1);
    if ((xenv[0]&0xffffff) != _neventcnt) {
      printf("event counter err [%x] != %x\n",
             xenv[0], _neventcnt);
      const uint32_t* eh = reinterpret_cast<const uint32_t*>(buffer);
      for(unsigned i=0; i<8; i++)
        printf(" %08x",eh[i]);
      printf("\n");
      dump(buffer,ret);
    }
    _neventcnt = xenv[0]&0xffffff;
  }
  _neventcnt = (_neventcnt+1)&0xffffff;

  _nevents++;
  _nbytes += ret;

  if (event_header->seq.isEvent()) {
    unsigned streams(buffer[26]>>4);
    const char* p = buffer+32;

    if (!streams) {
      printf("No streams!\n");
      if (!ldump)
        dump(buffer,ret);
    }

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

    if (p != buffer+ret) {
      printf("dma size mismatch ret [%x] iterate[%x]\n",
             ret, unsigned(p-buffer));
    }
  }
}

Fmc126Validator::Fmc126Validator(const Configuration& cfg,
                                 unsigned testpattern) :
  Validator(cfg)
{
}

bool Fmc126Validator::_validate_raw(const XtcData::Transition& transition,
                                    const StreamHeader&        s) {
  bool lret=false;
  // Decode stream header
  _revents++;
  _rbytes += s.num_samples()*2;

  if (_lverbose) {
    printf("raw header %04x:%04x %04x[%04x]\n",
           s._p[3]&0xffff,s._p[3]>>16,s.num_samples(),s.cache_len(8));
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

bool Fmc126Validator::_validate_fex(const StreamHeader&        s) {
  bool lret=false;
  // Decode stream header
  _fevents++;
  _fbytes += s.num_samples()*2;

  if (_lverbose) {
    printf("fex header %04x:%04x %04x[%04x]\n",
           s._p[3]&0xffff,s._p[3]>>16,s.num_samples(),s.cache_len(8));
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


static const uint16_t _test_pattern_5[] = {
  0xf01, 0xf01, 0xe11, 0xe11, 0xd21, 0xd21, 0xc31, 0xc31,
  0xf02, 0xf02, 0xe12, 0xe12, 0xd22, 0xd22, 0xc32, 0xc32,
  0xf03, 0xf03, 0xe13, 0xe13, 0xd23, 0xd23, 0xc33, 0xc33,
  0xf04, 0xf04, 0xe14, 0xe14, 0xd24, 0xd24, 0xc34, 0xc34,
  0xf05, 0xf05, 0xe15, 0xe15, 0xd25, 0xd25, 0xc35, 0xc35 };


Fmc134Validator::Fmc134Validator(const Configuration& cfg,
                                 unsigned testpattern) :
  Validator   (cfg),
  _testpattern(testpattern),
  _num_samples_raw(0)
{
}

bool Fmc134Validator::_validate_raw(const XtcData::Transition& transition,
                                    const StreamHeader&        s) {
  bool lret=false;
  // Decode stream header
  _revents++;
  _rbytes += s.num_samples()*2;

  const uint16_t* data = reinterpret_cast<const uint16_t*>(&s+1);

  if (_lverbose) {
    printf("raw header %04x:%04x %04x[%04x]\n",
           s._p[3]&0xffff,s._p[3]>>16,s.num_samples(),s.cache_len(10));
    for(unsigned i=0; i<s.num_samples(); i++)
      printf("%04x%c",data[i],(i&0xf)==0xf?'\n':' ');
    printf("\n");
  }

  //  Validate:
  //    Number of samples
  //    Cache length
  //    Test pattern
  if (s.num_samples() != s.cache_len(10)*4) {
    if ((_lverbose || _nprints)) {
      //      _nprints--;
      printf("Header mismatch num_samples[%x] != 4*cache_len[%x]\n",
             s.num_samples(), s.cache_len(10));
      if (!_lverbose)
        printf("raw header %04x:%04x %04x[%04x]\n",
               s._p[3]&0xffff,s._p[3]>>16,s.num_samples(),s.cache_len(10));
    }
    lret = true;
  }
  
  if (_num_samples_raw==0)
    _num_samples_raw = s.num_samples();
  else
    if (_num_samples_raw != s.num_samples()) {
      if ((_lverbose || _nprints)) {
        _nprints--;
        printf("Change in num_samples[%x] != [%x]\n",
               s.num_samples(), _num_samples_raw);
        lret = true;
      }
    }
  
  if (_testpattern==5) {
    //  Compare against previous event
    //  [Can't do until sample clock is locked to timing]
    if (_transition.env) {
    }
    //  Check internal consistency
    bool lerr = false;
    for(unsigned i=0; i<s.num_samples(); i++)
      if (data[i] != _test_pattern_5[i%(sizeof(_test_pattern_5)>>1)]) {
        if (!lerr) {
          lerr = true;
          _internal_sample_errors++;
          if ((_lverbose || _nprints)) {
            _nprints--;
            printf("Test pattern failed data[%04x] != pattern[%04x] @ %i\n",
                   data[i], _test_pattern_5[i%(sizeof(_test_pattern_5)>>1)], i);
            lret = true;
          }
        }
      }
  }

  _transition = transition;
  _sample_value = *data;

  return lret;
}

bool Fmc134Validator::_validate_fex(const StreamHeader&        s) {
  bool lret=false;
  // Decode stream header
  _fevents++;
  _fbytes += s.num_samples()*2;

  if (_lverbose) {
    printf("fex header %04x:%04x %04x[%04x]\n",
           s._p[3]&0xffff,s._p[3]>>16,s.num_samples(),s.cache_len(10));
  }

  //  Could validate against raw data

  return lret;
}

