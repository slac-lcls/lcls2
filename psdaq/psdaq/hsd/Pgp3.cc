#include "psdaq/hsd/Pgp3.hh"
#include "psdaq/mmhw/Pgp3Axil.hh"

using namespace Pds::HSD;

Pgp3::Pgp3(Pds::Mmhw::Pgp3Axil& axi) : _axi(axi) {}

void   Pgp3::resetCounts    () { _axi.countReset=1; usleep(10); _axi.countReset=0; };
void   Pgp3::loopback       (bool v) { _axi.loopback = v?2:0; }
void   Pgp3::skip_interval  (unsigned v) { _axi.skpInterval = v; }
bool   Pgp3::localLinkReady () const { return (_axi.rxStatus>>1)&1; }
bool   Pgp3::remoteLinkReady() const { return (_axi.rxStatus>>2)&1; }
double   Pgp3::txClkFreqMHz () const { return _axi.txClkFreq*1.e-6; }
double   Pgp3::rxClkFreqMHz () const { return _axi.rxClkFreq*1.e-6; }
unsigned Pgp3::txCount      () const { return _axi.txFrameCnt; }
unsigned Pgp3::txErrCount   () const { return _axi.txFrameErrCnt; }
unsigned Pgp3::rxOpCodeCount() const { return _axi.rxOpCodeCnt; }
unsigned Pgp3::rxOpCodeLast () const { return _axi.rxOpCodeLast; }
unsigned Pgp3::remPause     () const { return _axi.remRxOflow; }
bool     Pgp3::loopback     () const { return _axi.loopback; }

