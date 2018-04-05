/**
 **  Read actual data file and validate sparsification
 **/

#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <new>
#include "xtcdata/xtc/Dgram.hh"
#include "psdaq/hsd/hsd.hh"

using namespace Pds::HSD;

//
//  Validate raw stream : ramp signal repeats 0..0xfe
//      phyclk period is 0.8 ns 
//      recTimingClk period is 5.384 ns
//        => 1348 phyclks per beam period
//

static bool _interleave = false;

namespace Pds {
  namespace HSD {
class RawStream {
public:
  RawStream(const EventHeader& event, const StreamHeader& strm) :
    _adc(reinterpret_cast<const uint16_t*>(&strm+1)[strm.boffs()]),
    _pid(event.pulseId())
  {
  }
public:
  static void interleave(bool v) { _interleave=v; }

  bool validate(const EventHeader& event, const StreamHeader& next) const {
    uint16_t adc = adcVal(event.pulseId());
    unsigned i=next.boffs();
    unsigned nerror(0);
    unsigned ntest (0);
    const unsigned end = next.samples()-8+next.eoffs();
    const uint16_t* p = reinterpret_cast<const uint16_t*>(&next+1);
    if (p[i] != adc) {
        ++nerror;
        printf("=== ERROR: Mismatch at first sample: adc [%x]  expected [%x]  delta[%d]\n",
               p[i], adc, (p[i]-adc)&0x7ff);
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
    printf("DPID = %u\n", (pulseId-_pid)&0xffffffff);
    uint64_t dclks = (pulseId-_pid)*1348;
    //    unsigned adc = (_adc+dclks)%255;
    unsigned adc = (_adc+dclks)&0x7ff;
    if (_interleave)
      adc = (_adc+4*dclks)&0x7ff;
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
    const unsigned end = _strm.samples()-8+_strm.eoffs();
    const unsigned end_j = raw.samples()-8+raw  .eoffs();
    const uint16_t* p_thr = reinterpret_cast<const uint16_t*>(&_strm+1);
    const uint16_t* p_raw = reinterpret_cast<const uint16_t*>(&raw  +1);
    unsigned i=_strm.boffs(), j=raw.boffs();
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
}
}
    

extern int optind;

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -f <filename>\n");
  printf("         -m <fex minimum>\n");
  printf("         -M <fex maximum>\n");
  printf("         -s <stream mask>\n");
  printf("         -c (skip containers - DRP file)\n");
  printf("         -n (no fex)\n");
  printf("         -t (text file - simulation)\n");
  printf("         -i (interleave)\n");
}

int main(int argc, char** argv) {

  printf("Start\n");
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

  while ( (c=getopt( argc, argv, "f:m:M:s:cinht")) != EOF ) {
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
    case 'i':
      RawStream::interleave(true);
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
  RawStream* vraw=0;
  unsigned skipSize = 0x924;

  while(1) {  // event loop
    if (lText) {
      if ((sz=getline(&line, &linesz, f))<=0)
        break;
      printf("Readline %zd [%32.32s]\n",sz, line);
      char* p = line;
      for(unsigned i=0; i<(sz+3)/4; i++, p++)
        event[i] = strtoul(p, &p, 16);
    }

    const EventHeader& eh = *reinterpret_cast<const EventHeader*>(event);
    if (eh.eventType()) continue;

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
      printf("Starting raw\n");
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
      printf("Starting fex\n");
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
