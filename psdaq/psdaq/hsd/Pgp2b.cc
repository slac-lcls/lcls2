#include "psdaq/hsd/Pgp2b.hh"
#include "psdaq/mmhw/Pgp2bAxi.hh"

#include <stdint.h>
#include <stdio.h>

using namespace Pds::HSD;

Pgp2b::Pgp2b(Pds::Mmhw::Pgp2bAxi& axi) : _axi(axi) {}

bool   Pgp2b::localLinkReady () const { return (_axi._status>>2)&1; }
bool   Pgp2b::remoteLinkReady() const { return (_axi._status>>3)&1; }
double   Pgp2b::txClkFreqMHz () const { return _axi._txClkFreq*1.e-6; }
double   Pgp2b::rxClkFreqMHz () const { return _axi._rxClkFreq*1.e-6; }
unsigned Pgp2b::txCount      () const { return _axi._txFrames; }
unsigned Pgp2b::txErrCount   () const { return _axi._txFrameErrs; }
unsigned Pgp2b::rxOpCodeCount() const { return _axi._rxOpcodes; }
unsigned Pgp2b::rxOpCodeLast () const { return _axi._lastRxOpcode; }

