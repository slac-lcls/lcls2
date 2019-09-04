#include "psdaq/hsd/Pgp2b.hh"
#include "psdaq/mmhw/Pgp2bAxi.hh"

#include <stdint.h>
#include <stdio.h>

using namespace Pds::HSD;

Pgp2b::Pgp2b(Pds::Mmhw::Pgp2bAxi& axi) : _axi(axi) {}

void   Pgp2b::resetCounts    () { _axi._countReset=1; usleep(10); _axi._countReset=0; };
void   Pgp2b::loopback       (bool v) { _axi._loopback = v?2:0; }
bool   Pgp2b::localLinkReady () const { return (_axi._status>>2)&1; }
bool   Pgp2b::remoteLinkReady() const { return (_axi._status>>3)&1; }
unsigned Pgp2b::remoteLinkId () const { return 0; }
double   Pgp2b::txClkFreqMHz () const { return _axi._txClkFreq*1.e-6; }
double   Pgp2b::rxClkFreqMHz () const { return _axi._rxClkFreq*1.e-6; }
unsigned Pgp2b::txCount      () const { return _axi._txFrames; }
unsigned Pgp2b::txErrCount   () const { return _axi._txFrameErrs; }
unsigned Pgp2b::rxOpCodeCount() const { return _axi._rxOpcodes; }
uint64_t Pgp2b::rxOpCodeLast () const { return _axi._lastRxOpcode; }
unsigned Pgp2b::remPause     () const { return _axi._remoteOvfVc0; }
bool     Pgp2b::loopback     () const { return _axi._loopback; }

