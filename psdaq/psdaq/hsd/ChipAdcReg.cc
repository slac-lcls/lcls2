#include "psdaq/hsd/ChipAdcReg.hh"

#include <unistd.h>
#include <stdio.h>

using namespace Pds::HSD;

void ChipAdcReg::init()
{
  unsigned v = csr;
  v &= ~(1<<31);
  csr = v | (1<<4);
  usleep(10);
  csr = v & ~(1<<4);
}

void ChipAdcReg::start()
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

void ChipAdcReg::stop()
{
  unsigned v = csr;
  v &= ~(1<<31);
  v &= ~(1<<1);
  csr = v;
}

void ChipAdcReg::resetCounts()
{
  unsigned v = csr;
  csr = v | (1<<0);
  usleep(10);
  csr = v & ~(1<<0);
}

void ChipAdcReg::setChannels(unsigned ch)
{
  unsigned v = control;
  v &= ~0xff;
  v |= (ch&0xff);
  control = v;
}

void ChipAdcReg::setupDaq(unsigned partition)
{
  //  acqSelect = (1<<30) | (3<<11) | partition;  // obsolete
  // { unsigned v = control;
  //   v &= ~(0xff << 16);
  //   v |= (partition&0xf) << 16;
  //   v |= (partition&0xf) << 20;
  //   control = v; }
  unsigned v = csr & ~(1<<0);
  csr = v | (1<<0);
}

void ChipAdcReg::resetClock(bool r)
{
  unsigned v = csr;
  if (r) 
    v |= (1<<3);
  // Self clearing (synchronous to EVR strobe)
  // else
  //   v &= ~(1<<3);
  csr = v;
}

void ChipAdcReg::resetDma()
{
  unsigned v = csr;
  v |= (1<<4);
  csr = v;
  usleep(10);
  v &= ~(1<<4);
  csr = v;
}

void ChipAdcReg::resetFb()
{
  unsigned v = csr;
  v |= (1<<5);
  csr = v;
  usleep(10);
  v &= ~(1<<5);
  csr = v;
  usleep(10);
}

void ChipAdcReg::resetFbPLL()
{
  unsigned v = csr;
  v |= (1<<6);
  csr = v;
  usleep(10);
  v &= ~(1<<6);
  csr = v;
  usleep(10);
}

void ChipAdcReg::setLocalId(unsigned v) 
{ 
  //  localId = v; 
  printf("*** ChipAdcReg::setLocalId deprecated ***\n");
}

void ChipAdcReg::dump() const
{
#define PR(r) printf("%9.9s: %08x\n",#r, r)

  //  PR(localId);
  //  PR(upstreamId);
  //  PR(dnstreamId[0]);
  PR(irqEnable);
  PR(irqStatus);
  //  PR(partitionAddr);
  PR(dmaFullThr);
  PR(csr);
  //  PR(acqSelect);
  PR(control);
  PR(samples);
  PR(prescale);
  PR(offset);
  PR(countAcquire);
  PR(countEnable);
  PR(countInhibit);
  PR(countRead);
  //  PR(countStart);
  //  PR(countQueue);

#undef PR
}
