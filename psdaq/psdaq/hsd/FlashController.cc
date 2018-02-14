#include "psdaq/hsd/FlashController.hh"

#include <ctype.h>
#include <unistd.h>

using namespace Pds::HSD;

void FlashController::write(const uint32_t* p, unsigned nwords)
{
  _command = 1; // reset
  usleep(10);
  _command = 0;
  usleep(250000);
  
  _bytes_to_prog = nwords*sizeof(uint32_t);

  for(unsigned i=0; i<nwords;) {
    // printf("Writing word %08x\n",*p);
    _data = *p++;
    i++;
    if ((i%(nwords/10))==0)
      printf("%u%% complete [%lu B]\n",100*i/nwords, i*sizeof(uint32_t));
    else if ((i%(nwords/100))==0) {
      if ((i%(nwords/10))==(nwords/100))
        printf("Writing");
      else
        printf(".");
    }
    if ((i%2048)==2047) {
      usleep(200000);
    }
  }
  printf("\n");
}

void FlashController::write(FILE* f)
{
  char* d = new char[1<<28];
  unsigned i=0;
  int c;
  while( (c=fgetc(f))!= EOF ) {
    d[i>>1] <<= 4;
    if (isdigit(c)) 
      d[i>>1] |= c-'0';
    else if (isxdigit(c))
      d[i>>1] |= c-'a'+10;
    else {
      printf("Unexpected character %u [%c] at %u\n",
             c, (char)c, i);
      break;
    }
    if ((i++&0x1fffff)==0x1fffff)
      printf("%u MB\n",i>>21);
  }

  //  Round up number of words to the nearest whole page
  //    Fractional page readback isn't working, I guess.
  unsigned nw = (i+7)>>3;
  nw = (nw+0x1f)&~0x1f;

  printf("Writing %u words\n", nw);
  write(reinterpret_cast<const uint32_t*>(d), nw);
  usleep(1000000);
  read (reinterpret_cast<const uint32_t*>(d), nw);
  delete[] d;
}

int  FlashController::read(const uint32_t* p, unsigned nwords)
{
  const uint16_t* q = reinterpret_cast<const uint16_t*>(p);

#if 0
  _command = 1; // reset
  usleep(10);
  _command = 0;
  usleep(250000);
#endif
  
  _bytes_to_read = nwords*sizeof(uint32_t);

  nwords <<= 1;
  //  unsigned vold = 0;
  for(unsigned i=0; i<nwords; ) {
    unsigned v = _data;
    //  if (v!=vold || (v&(1<<31)))
    //    printf("Read word %08x\n",v);
    if (v>>31) {
      uint16_t data = v&0xffff;
      if (*q != data) {
        printf("\nVerify failed [%04x:%04x] at %lu\n",
               data,*q,q-reinterpret_cast<const uint16_t*>(p));
        break;
      }
      q++;
      i++;
      if ((i%(nwords/10))==0)
        printf("Verify %u%% complete [%lu B]\n",100*i/nwords, i*sizeof(uint16_t));
    }
    else {
      // if (i*10>9*nwords && v!=vold)
      //   printf("[%u B]\n",i*sizeof(uint16_t));
      usleep(10);
    }
    //    vold=v;
  }
  printf("\n");

  return (q-reinterpret_cast<const uint16_t*>(p))>>1;
}

