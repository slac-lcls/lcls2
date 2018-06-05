#include "psalg/include/stream.hh"
#include "psalg/include/hsd.hh"

#include <stdio.h>
#include <ctype.h>

using namespace Pds::HSD;

static bool _interleave = false;
static unsigned _lverbose = 0;

RawStream::RawStream(const EventHeader& event, const StreamHeader& strm) :
  _pid  (event.pulseId()),
  _adc  (reinterpret_cast<const uint16_t*>(&strm+1)[strm.boffs()]),
  _baddr(strm.baddr()),
  _eaddr(strm.eaddr())
{
}

void RawStream::interleave(bool v)     { _interleave=v; }
void RawStream::verbose   (unsigned v) { _lverbose  =v; }

bool RawStream::validate(const EventHeader& event, const StreamHeader& next) const {
  if (next.samples()==0) return true;
  uint16_t adc = adcVal(event.pulseId());
  unsigned i=next.boffs();
  unsigned nerror(0);
  unsigned ntest (0);
  const unsigned end = next.samples()-next.eoffs();
  const uint16_t* p = reinterpret_cast<const uint16_t*>(&next+1);
  if (p[i] != adc) {
    ++nerror;
    if (_lverbose) {
      printf("DPID = %" PRIu64 "\n", (event.pulseId()-_pid)&0xffffffff);
      printf("=== ERROR: Mismatch at first sample: adc [%x]  expected [%x]  delta[%d]\n",
             p[i], adc, (p[i]-adc)&0x7ff);
    }    
  }
  ntest++;
  adc = this->next(p[i]);

  i++;
  while(i<end) {
    ntest++;
    if (p[i] != adc) {
      ++nerror;
      if (_lverbose && nerror < 10) {
        if (nerror==1)
          printf("DPID = %" PRIu64 "\n", (event.pulseId()-_pid)&0xffffffff);
        printf("=== ERROR: Mismatch at index %u : adc [%x]  expected [%x]\n",
               i, p[i], adc);
      }
    }
    adc = this->next(p[i]);
    i++;
  }

  if (_lverbose>1) 
    printf("RawStream::validate %u/%u errors\n", nerror, ntest);

  return nerror==0;
}

unsigned RawStream::adcVal(uint64_t pulseId) const {
  uint64_t dclks = (pulseId-_pid)*1348;
  unsigned adc = (_adc+dclks)&0x7ff;
  if (_interleave)
    adc = (_adc+4*dclks)&0x7ff;
  return adc;
}

uint16_t RawStream::next(uint16_t adc) const {
  return (adc+1)&0x7ff;
}

ThrStream::ThrStream(const StreamHeader& strm) :
  _strm(strm)
{
}

bool ThrStream::validate(const StreamHeader& raw) const {
  //  (1) Generate a compressed stream from the raw stream and compare, or
  //  (2) Verify each word of the compressed stream is found in the raw stream at the right location

  if (_strm.samples()==0 || raw.samples()==0) return true;
  unsigned nerror(0), ntest(0);
  const unsigned end = _strm.samples()-_strm.eoffs();
  const unsigned end_j = raw.samples()-raw  .eoffs();
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
        if (_lverbose && (nerror < 10))
          printf("=== ERROR: Mismatch at index thr[%u], raw[%u] : adc thr[%x] raw[%x]\n",
                 i, j, p_thr[i], p_raw[j]);
      }
      j++;
    }
    i++;
  }
  printf("nerror/ntest: %d %d\n", nerror, ntest);
  return nerror==0;
}
