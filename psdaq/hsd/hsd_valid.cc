/**
 **  Read actual data file and validate sparsification
 **/

#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <new>

class StreamHeader {
public:
  StreamHeader() {}
public:
  unsigned samples () const { return _word[0]&0x7fffffff; }
  bool     overflow() const { return _word[0]>>31; }
  unsigned boffs   () const { return (_word[1]>>0)&0xff; }
  unsigned eoffs   () const { return (_word[1]>>8)&0xff; }
  unsigned buffer  () const { return _word[1]>>16; }
  unsigned toffs   () const { return _word[2]; }
  unsigned baddr   () const { return _word[3]&0xffff; }
  unsigned eaddr   () const { return _word[3]>>16; }
  void     dump    () const
  {
    printf("  ");
    for(unsigned i=0; i<4; i++)
      printf("%08x%c", _word[i], i<3 ? '.' : '\n');
    printf("  size [%04u]  boffs [%u]  eoffs [%u]  buff [%u]  toffs[%04u]  baddr [%04x]  eaddr [%04x]\n",
           samples(), boffs(), eoffs(), buffer(), toffs(), baddr(), eaddr());
  }
private:
  unsigned _word[4];
};

class EventHeader {
public:
  EventHeader() {}
public:
  uint64_t pulseId   () const { return *reinterpret_cast<const uint64_t*>(&_word[0]); }
  uint64_t timeStamp () const { return *reinterpret_cast<const uint64_t*>(&_word[2]); }
  uint16_t trigWord  () const { return reinterpret_cast<const uint16_t*>(&_word[4])[1]; }
  uint32_t eventCount() const { return _word[5]; }
  unsigned samples   () const { return _word[6]&0x3ffff; }
  unsigned channels  () const { return _word[6]>>24; }
  unsigned sync      () const { return _word[7]&0x7; }

  void dump() const 
  {
    for(unsigned i=0; i<8; i++)
      printf("%08x%c", _word[i], i<7 ? '.' : '\n');
    printf("pID [%016llx]  time [%u.%09u]  trig [%04x]  event [%u]  sync [%u]\n",
           (unsigned long long)pulseId(), _word[3], _word[2], trigWord(), eventCount(), sync());
  }
private:
  unsigned _word[8];
};

//
//  Validate raw stream : ramp signal repeats 0..0xfe
//      phyclk period is 0.8 ns 
//      recTimingClk period is 5.384 ns
//        => 1348 phyclks per beam period
//

class RawStream {
public:
  RawStream(const EventHeader& event, const StreamHeader& strm) :
    _adc(reinterpret_cast<const uint16_t*>(&strm+1)[strm.boffs()]),
    _pid(event.pulseId())
  {
  }
public:
  bool validate(const EventHeader& event, const StreamHeader& next) const {
    uint16_t adc = adcVal(event.pulseId());
    unsigned i=next.boffs();
    unsigned nerror(0);
    unsigned ntest (0);
    const unsigned end = next.samples()-8 + next.eoffs();
    const uint16_t* p = reinterpret_cast<const uint16_t*>(&next+1);
    if (p[i] != adc) {
        ++nerror;
        printf("=== ERROR: Mismatch at first sample: adc [%x]  expected [%x]  delta[%d]\n",
               p[i], adc, p[i]-adc);
    }
    adc = this->next(p[i]);

    i++;
    while(i<end) {
      ntest++;
      if (p[i] != adc) {
        ++nerror;
        if (nerror < 10)
          printf("=== ERROR: Mismatch at index %u : adc [%x]  expected [%x]\n",
                 i, p[i], adc);
      }
      adc = this->next(p[i]);
      i++;
    }
    printf("RawStream::validate %u/%u errors\n", nerror, ntest);
    return nerror==0;
  }
private:
  unsigned adcVal(uint64_t pulseId) const {
    uint64_t dclks = (pulseId-_pid)*1348;
    //    unsigned adc = (_adc+dclks)%255;
    unsigned adc = (_adc+dclks)&0x7ff;
    return adc;
  }
  uint16_t next(uint16_t adc) const {
    //    return (adc+1)%255;
    return (adc+1)&0x7ff;
  }
private:
  unsigned _adc;
  uint64_t _pid;
};
    

//
//  Validate threshold stream : ramp signal repeats 0..0xfe
//      phyclk period is 0.8 ns 
//      recTimingClk period is 5.384 ns
//        => 1346 phyclks per beam period
//

class ThrStream {
public:
  ThrStream(const StreamHeader& strm) :
    _strm(strm)
  {
  }
public:
  bool validate(const StreamHeader& raw) const {
    //  (1) Generate a compressed stream from the raw stream and compare, or
    //  (2) Verify each word of the compressed stream is found in the raw stream at the right location

    unsigned nerror(0), ntest(0);
    const unsigned end = _strm.samples()-8 + _strm.eoffs() - _strm.boffs();
    const unsigned end_j = raw.samples()-8 + raw  .eoffs() - raw  .boffs();
    const uint16_t* p_thr = &reinterpret_cast<const uint16_t*>(&_strm+1)[_strm.boffs()];
    const uint16_t* p_raw = &reinterpret_cast<const uint16_t*>(&raw  +1)[raw  .boffs()];
    unsigned i=0, j=0;
    if (p_thr[i] & 0x8000) { // skip to the sample with the trigger
      i++;
      j++;
    }
    while(i<end && j<end_j) {
      if (p_thr[i] & 0x8000) {  // skip
        j += p_thr[i] & 0x7fff;
      }
      else {
        ntest++;
        if (p_thr[i] != p_raw[j]) {
          nerror++;
          if (nerror < 10)
            printf("=== ERROR: Mismatch at index thr[%u], raw[%u] : adc thr[%x] raw[%x]\n",
                   i, j, p_thr[i], p_raw[j]);
        }
        j++;
      }
      i++;
    }

    printf("ThrStream::validate %u/%u errors\n", nerror, ntest);
    return nerror==0;
  }
private:
  const StreamHeader& _strm;
};
    

extern int optind;

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -f <filename>\n");
}

int main(int argc, char** argv) {

  extern char* optarg;
  char* endptr;

  int c;
  bool lUsage = false;
  const char* fname = 0;
  unsigned streams = 3;
  unsigned FMIN=0x41, FMAX=0x3bf;
  bool lNoFex = false;
  bool lText = false;
  bool lSkipContainers = false;

  while ( (c=getopt( argc, argv, "f:m:M:s:cnht")) != EOF ) {
    switch(c) {
    case 'c':
      lSkipContainers = true;
      break;
    case 'f':
      fname = optarg;
      break;
    case 'm':
      FMIN = strtoul(optarg,&endptr,0);
      break;
    case 'M':
      FMAX = strtoul(optarg,&endptr,0);
      break;
    case 's':
      streams = strtoul(optarg, &endptr, 0);
      break;
    case 't':
      lText = true;
      break;
    case 'n':
      lNoFex = true;
      break;
    case '?':
    default:
      lUsage = true;
      break;
    }
  }

  if (lUsage) {
    usage(argv[0]);
    exit(1);
  }

  FILE* f = fopen(fname,"r");
  if (!f) {
    perror("Unable to open input file");
    return -1;
  }

  uint32_t* event = new uint32_t[0x100000];
  uint16_t* compr = new uint16_t[0x1000];

  size_t linesz = 0x10000;
  char* line = new char[linesz];
  ssize_t sz;
  unsigned ievent=0;
  RawStream* vraw=0;
  unsigned skipSize = 0x924;

  while(1) {  // event loop
    if (lText) {
      if ((sz=getline(&line, &linesz, f))<=0)
        break;
      printf("Readline %d [%32.32s]\n",sz, line);
      char* p = line;
      for(unsigned i=0; i<(sz+3)/4; i++, p++)
        event[i] = strtoul(p, &p, 16);
    }

    const EventHeader& eh = *reinterpret_cast<const EventHeader*>(event);
    //    printf("Read event header into %p\n", &eh);
    if (!lText) {
      if (lSkipContainers)
        if (fread((void*)&eh, skipSize, 1, f) == 0)
          return 0;
      if (fread((void*)&eh, sizeof(EventHeader), 1, f) == 0)
        return 0;
      skipSize = 0xc8;
    }
    eh.dump();

    const char* next = reinterpret_cast<const char*>(&eh+1);

    const StreamHeader* sh_raw = 0;
    if (streams&1) {
      sh_raw = reinterpret_cast<const StreamHeader*>(next);
      if (!lText) {
        if (fread((void*)sh_raw, sizeof(StreamHeader), 1, f) == 0)
          break;
      }
      sh_raw->dump();

      const uint16_t* raw = reinterpret_cast<const uint16_t*>(sh_raw+1);
      if (!lText) {
        if (fread((void*)raw, 2, sh_raw->samples(), f) == 0)
          break;
      }
      printf("\t"); for(unsigned i=0; i<8; i++) printf(" %04x", raw[i]); printf("\n");

      if (!vraw)
        vraw = new RawStream(eh, *sh_raw);
      vraw->validate(eh, *sh_raw);

      next = reinterpret_cast<const char*>(&raw[sh_raw->samples()]);
    }

    if (streams&2) {
      const StreamHeader& sh_fex = *reinterpret_cast<const StreamHeader*>(next);
      //    printf("Read header into %p\n",&sh_fex);
      if (!lText) {
        if (fread((void*)&sh_fex, sizeof(StreamHeader), 1, f) == 0)
          break;
      }
      sh_fex.dump();

      const uint16_t* fex = reinterpret_cast<const uint16_t*>(&sh_fex+1);
      if (!lText) {
        if (fread((void*)fex, 2, sh_fex.samples(), f) == 0)
          break;
      }
      printf("\t"); for(unsigned i=0; i<8; i++) printf(" %04x", fex[i]); printf("\n");

      if (!lNoFex && sh_raw) {
        ThrStream vthr(sh_fex);
        vthr.validate(*sh_raw);
      }
    }
    //    return 0;

    printf("-----\n");
  }
  return 0;

  {
    const uint32_t* f = reinterpret_cast<const uint32_t*>(&event[8]);
    const uint16_t* s = reinterpret_cast<const uint16_t*>(&event[12]);
    uint16_t* c = compr;
    int toffs=-1;
    { 
      const uint16_t* s_beg = s + (f[1]&0xff);
      const uint16_t* s_end = s + (f[0]-8)+((f[1]>>8)&0xff)-1;
      printf("\tRaw: %08x.%08x.%08x.%08x\n", f[0], f[1], f[2], f[3]);
      printf("\t\t%04x.%04x [%ld]\n",*s_beg,*s_end,(s_end-s_beg)+1);

      //  Compress the raw data
      int skip=-1;
      for(const uint16_t* q = s_beg; q<=s_end; q++) {
        if (*q < FMIN || *q > FMAX) {
          if (toffs<0)
            toffs = q-s_beg;
          if (skip>=0) {
            *c++ = 0x8000 | (skip&0x3fff);
            skip = -1;
          }
          *c++ = *q;
        }
        else if (toffs>=0) {
          skip++;
        }
      }
    }

    s += event[8];
    f = reinterpret_cast<const uint32_t*>(s);
    printf("\tFex: %08x.%08x.%08x.%08x\n", f[0], f[1], f[2], f[3]);

    if (f[0]) {
      s += 8; // skip header
      const uint16_t* s_beg = s + (f[1]&0xff);
      const uint16_t* s_end = s + (f[0]-8)+((f[1]>>8)&0xff)-1;
      uint16_t fex_beg = *s_beg;
      uint16_t fex_end = *s_end;
      printf("\t\t%04x.%04x [%ld]\n", fex_beg, fex_end, (s_end-s_beg)+1);
      
      //  Compare the header
      if (unsigned(toffs) != f[2])
        printf("\t\t\ttoffs[%u]  header[%u]\n", toffs, f[2]);

      if (unsigned(c-compr) != s_end-s_beg+1) 
        printf("\t\t\tfexl[%ld] cmpl[%ld]\n",
               s_end-s_beg+1, c-compr );

      //  Compare the words
      bool lErr=false;
      c = compr;
      int skipRem = 0;
      for(s = s_beg; s <= s_end; s++, c++) {
        //  handle consecutive skip characters carefully
        while ((*s>>15)==1) {
          skipRem += (*s&0x3fff)+1;
          s++;
          if (s >= s_end)
            break;
        }

        while ((*c>>15)==1) {
          skipRem -= (*c&0x3fff)+1;
          c++;
        }

        if (skipRem)
          printf("\t\tSkip sequences disagree %d @ %ld\n", skipRem, s-s_beg);

        if (*s != *c) {
          printf("\t\t\tfex[%04x] cmp[%04x] @ %ld\n",
                 *s, *c, s-s_beg);
          lErr=true;
        }
      }

      if (lErr) {
        for(unsigned i=0; i<=(s_end-s_beg); i+=4) {
          printf("\t\t%04x %04x %04x %04x   %04x %04x %04x %04x  [%u]\n",
                 s_beg[i+0],s_beg[i+1],s_beg[i+2],s_beg[i+3],
                 compr[i+0],compr[i+1],compr[i+2],compr[i+3], i);
        }
      }
    }
  }

  return 1;
}
