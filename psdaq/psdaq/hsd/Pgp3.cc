#include "psdaq/hsd/Pgp3.hh"
#include "psdaq/mmhw/Pgp3Axil.hh"

using namespace Pds::HSD;

#define MASK(nb) ((1<<nb)-1)
#define SETFIELD(r,v,b,nb) { unsigned w = r; w&=~(MASK(nb)<<b); w|=(v&MASK(nb))<<b; r = w; }

Pgp3::Pgp3(Pds::Mmhw::Pgp3Axil& axi) : _axi(axi) {}

void   Pgp3::resetCounts    () { _axi.countReset=1; usleep(10); _axi.countReset=0; };
void   Pgp3::loopback       (bool v) { _axi.loopback = v?2:0; }
void   Pgp3::skip_interval  (unsigned v) { _axi.skpInterval = v; }
void   Pgp3::txdiffctrl     (unsigned v) { SETFIELD(_axi.txGthDriver,v,0,8); }
void   Pgp3::txprecursor    (unsigned v) { SETFIELD(_axi.txGthDriver,v,8,8); }
void   Pgp3::txpostcursor   (unsigned v) { SETFIELD(_axi.txGthDriver,v,16,8); }

bool   Pgp3::localLinkReady () const { return (_axi.rxStatus>>1)&1; }
bool   Pgp3::remoteLinkReady() const { return (_axi.rxStatus>>2)&1; }
unsigned Pgp3::remoteLinkId () const { return (rxOpCodeLast()>>16)&0xffffffff; }
double   Pgp3::txClkFreqMHz () const { return _axi.txClkFreq*1.e-6; }
double   Pgp3::rxClkFreqMHz () const { return _axi.rxClkFreq*1.e-6; }
unsigned Pgp3::txCount      () const { return _axi.txFrameCnt; }
unsigned Pgp3::txErrCount   () const { return _axi.txFrameErrCnt; }
unsigned Pgp3::rxCount      () const { return _axi.rxFrameCnt; }
unsigned Pgp3::rxErrCount   () const { return _axi.rxFrameErrCnt; }
unsigned Pgp3::rxOpCodeCount() const { return (_axi.rxOpCodeCnt>>24); }
uint64_t Pgp3::rxOpCodeLast () const 
{
  uint64_t v = _axi.rxOpCodeNum&0xffff;
  v <<= 32;
  v |= _axi.rxOpCodeLast;
  return v;
}

unsigned Pgp3::remPause     () const { return _axi.remRxOflow; }
bool     Pgp3::loopback     () const { return _axi.loopback; }

