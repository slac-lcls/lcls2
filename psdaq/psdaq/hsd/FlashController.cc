#include "psdaq/hsd/FlashController.hh"
#include "psdaq/mmhw/HexFile.hh"
#include "psdaq/mmhw/McsFile.hh"

#include <ctype.h>
#include <unistd.h>
#include <string.h>

using namespace Pds::HSD;

static bool lverbose = false;
static bool luseFifo = true;

void FlashController::verbose(bool v) {lverbose=v;}
void FlashController::useFifo(bool v) {luseFifo=v;}

void FlashController::_write(const unsigned* p, unsigned nwords)
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

void FlashController::_write(const std::vector<uint8_t>& data)
{
  const unsigned FIFO_AFULL = 8180; // 8192 full

  _command = 1; // reset
  usleep(10);
  _command = 0;
  usleep(250000);
  
  _bytes_to_prog = data.size();
  unsigned nwords = (data.size()+3)>>2;

  unsigned nfifo = _prog_fifo_cnt;

  printf("Write %u words; Fifo count %u\n", nwords, nfifo);

  unsigned v=0;
  unsigned i;
  for(i=0; i<data.size(); i++) {
    v >>= 8;
    v |= unsigned(data[i])<<24;
    if ((i&3)==3) {
#if 1
      if (nfifo > FIFO_AFULL) {
        if (luseFifo) {
          nfifo = _prog_fifo_cnt;
          //        printf("nfifo at 0x%x\n",nfifo);
          while (nfifo > FIFO_AFULL) {
          usleep(10);
          nfifo = _prog_fifo_cnt;
          }
        }
        else {
          usleep(500000);
          nfifo = 0;
        }
      }
      nfifo+=2;
#endif
      _data = v;
      unsigned iw = i>>2;
      if (lverbose) {
        printf("Write word %u: %08x\n", iw,v);
      }
      if ((iw%(nwords/20))==0)
        printf("%u%% complete [%u B]\n",100*iw/nwords, i);
#if 0
      if ((iw%2048)==2047) {
        usleep(200000);
      }
#endif
    }
  }
  printf("\n");
}

void FlashController::write(const char* fname)
{
  const char* extp = strrchr(fname,'.');
  if (!extp) {
    printf("No file extension\n");
    return;
  }

  if (strcmp(extp,".hex")==0) {
    Pds::Mmhw::HexFile f(fname);
    _write(f.data());
  }
  else if (strcmp(extp,".mcs")==0) {
    Pds::Mmhw::McsFile m(fname);
    Pds::Mmhw::HexFile f(m);
    _write(f.data());
  }
}


int  FlashController::_verify(const unsigned* p, unsigned nwords)
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

int  FlashController::_verify(const std::vector<uint8_t>& d)
{
  _command = 1; // reset
  usleep(10);
  _command = 0;
  usleep(250000);

  _bytes_to_read  = d.size();
  unsigned nwords = d.size()>>1;

  for(unsigned i=0; i<nwords; ) {
    unsigned q = (d[(i<<1)+1]<<8) | d[(i<<1)];
    unsigned v = _data;
    if (v>>31) {
      uint16_t data = v&0xffff;
      if (lverbose)
        printf("[%04x:%04x] at %u\n",data,q,i);

      if (q != data) {
        printf("\nVerify failed [%04x:%04x] at %u\n",
               data,q,i);
        break;
      }

      i++;
      if ((i%(nwords/10))==0)
        printf("Verify %u%% complete [%lu B]\n",100*i/nwords, i*sizeof(uint16_t));
    }
    else {
      usleep(10);
    }
  }
  printf("\n");

  return nwords;
}

void FlashController::verify(const char* fname)
{
  const char* extp = strrchr(fname,'.');
  if (!extp) {
    printf("No file extension\n");
    return;
  }

  if (strcmp(extp,".hex")==0) {
    Pds::Mmhw::HexFile f(fname);
    _verify(f.data());
  }
  else if (strcmp(extp,".mcs")==0) {
    Pds::Mmhw::McsFile m(fname);
    Pds::Mmhw::HexFile f(m);
    _verify(f.data());
  }
}

void FlashController::_read(std::vector<uint8_t>& v, unsigned nwords)
{
  _command = 1; // reset
  usleep(10);
  _command = 0;
  usleep(250000);

  _bytes_to_read  = nwords<<1;

  for(unsigned i=0; i<nwords; ) {
    unsigned d = _data;
    if (d>>31) {
      v.push_back(d&0xff);
      v.push_back((d>>8)&0xff);

      i++;
      if ((i%(nwords/10))==0)
        printf("Read %u%% complete [%lu B]\n",100*i/nwords, i*sizeof(uint16_t));
    }
    else {
      usleep(10);
    }
  }
  printf("\n");
}

//  Write a text file of flash contents
std::vector<uint8_t> FlashController::read(unsigned nwords)
{
  std::vector<uint8_t> v;
  _read(v,nwords);
  return v;
}
