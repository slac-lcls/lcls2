/**
 **  Read simulated data file and validate sparsification
 **/

#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <new>

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
  unsigned FMIN=0x41, FMAX=0x3bf;

  while ( (c=getopt( argc, argv, "f:m:M:h")) != EOF ) {
    switch(c) {
    case 'f':
      fname = optarg;
      break;
    case 'm':
      FMIN = strtoul(optarg,&endptr,0);
      break;
    case 'M':
      FMAX = strtoul(optarg,&endptr,0);
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

  const unsigned linesz = 40960;
  char* line = new char[linesz];

  while(1) {  // event loop
    if ( !fgets(line, linesz, f) )
      break;

    uint32_t* u = event;
    char* p = line;
    while(*p) {
      *u++ = strtoul(p,&endptr,16);
      p = endptr+1;
    }

    printf("Event: %08x.%08x %08x.%08x\n",
           event[0], event[1], event[2], event[3]);

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
