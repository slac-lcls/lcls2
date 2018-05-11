#include "psdaq/hsd/QABase.hh"

#include <unistd.h>
#include <stdio.h>

using namespace Pds::HSD;

void QABase::init()
{
  unsigned v = csr;
  v &= ~(1<<31);
  csr = v | (1<<4);
  usleep(10);
  csr = v & ~(1<<4);
}

void QABase::start()
{
  unsigned v = control;
  v &= ~(1<<20);  // remove inhibit
  control = v;

  v = csr;
  //  csr = v | (1<<31) | (1<<1);
  v &= ~(1<<4);   // remove reset
  csr = v | (1<<31);

  irqEnable = 1;
}

void QABase::stop()
{
  unsigned v = csr;
  csr = v & ~(1<<31) & ~(1<<1);
}

void QABase::resetCounts()
{
  unsigned v = csr;
  csr = v | (1<<0);
  usleep(10);
  csr = v & ~(1<<0);
}

void QABase::setChannels(unsigned ch)
{
  unsigned v = control;
  v &= ~0xff;
  v |= (ch&0xff);
  control = v;
}

void QABase::setMode (Interleave q)
{
  unsigned v = control;
  if (q) v |=  (1<<8);
  else   v &= ~(1<<8);
  control = v;
}

void QABase::setupDaq(unsigned partition)
{
  acqSelect = (1<<30) | (3<<11) | partition;  // obsolete
  { unsigned v = control;
    v &= ~(0xf << 16);
    v |= (partition&0xf) << 16;
    control = v; }
  unsigned v = csr & ~(1<<0);
  csr = v | (1<<0);
}

void QABase::setupLCLS(unsigned rate)
{
  acqSelect = rate&0xff;
  unsigned v = csr & ~(1<<0);
  csr = v | (1<<0);
}

void QABase::setupLCLSII(unsigned rate)
{
  acqSelect = (1<<30) | (0<<11) | rate;
  unsigned v = csr & ~(1<<0);
  csr = v | (1<<0);
}

void QABase::setTrigShift(unsigned shift)
{
  unsigned v = csr & ~0xff00;
  csr = v | ((shift & 0xff)<<8);
}

void QABase::enableDmaTest(bool enable)
{
  unsigned v = csr;
  if (enable)
    v |= (1<<2);
  else
    v &= ~(1<<2);
  csr = v;
}

void QABase::resetClock(bool r)
{
  unsigned v = csr;
  if (r) 
    v |= (1<<3);
  else
    v &= ~(1<<3);
  csr = v;
}

void QABase::resetDma()
{
  unsigned v = csr;
  v |= (1<<4);
  csr = v;
  usleep(10);
  v &= ~(1<<4);
  csr = v;
}

void QABase::resetFb()
{
  unsigned v = csr;
  v |= (1<<5);
  csr = v;
  usleep(10);
  v &= ~(1<<5);
  csr = v;
  usleep(10);
}

void QABase::resetFbPLL()
{
  unsigned v = csr;
  v |= (1<<6);
  csr = v;
  usleep(10);
  v &= ~(1<<6);
  csr = v;
  usleep(10);
}

void QABase::dump() const
{
#define PR(r) printf("%9.9s: %08x\n",#r, r)

  PR(irqEnable);
  PR(irqStatus);
  PR(partitionAddr);
  PR(dmaFullThr);
  PR(csr);
  PR(acqSelect);
  PR(control);
  PR(samples);
  PR(prescale);
  PR(offset);
  PR(countAcquire);
  PR(countEnable);
  PR(countInhibit);
  PR(countRead);
  PR(countStart);
  PR(countQueue);

#undef PR
}
