#include "psdaq/hsd/FmcCore.hh"

using namespace Pds::HSD;

bool FmcCore::present() const
//{ return (_detect&1)==0; }
{ return powerGood(); }

bool FmcCore::powerGood() const
{ return _detect&2; }

void FmcCore::selectClock(unsigned i)
{ _clock_select = i; }

double FmcCore::clockRate() const
{ return double(_clock_count)/8192.*125.e6; }

void FmcCore::cal_enable()
{ _cmd |= (1<<0); }

void FmcCore::cal_disable()
{ _cmd &= ~(1<<0); }
