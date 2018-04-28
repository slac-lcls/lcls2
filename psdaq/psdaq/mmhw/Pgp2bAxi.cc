#include "psdaq/mmhw/Pgp2bAxi.hh"

#include <stdio.h>

using namespace Pds::Mmhw;

void Pgp2bAxi::dump() const
{
  printf("\tloopback       : %x\n",_loopback);
  printf("\ttxUserData     : %u\n",_txUserData);
  printf("\trxPhyReady     : %u\n",(_status>>0)&1);
  printf("\ttxPhyReady     : %x\n",(_status>>1)&1);
  printf("\tlocalLinkReady : %x\n",(_status>>2)&1);
  printf("\tremoteLinkReady: %x\n",(_status>>3)&1);
  printf("\ttransmitReady  : %x\n",(_status>>4)&1);
  printf("\trxPolarity     : %x\n",(_status>>8)&3);
  printf("\tremotePause    : %x\n",(_status>>12)&0xf);
  printf("\tlocalPause     : %x\n",(_status>>16)&0xf);
  printf("\tremoteOvfl     : %x\n",(_status>>20)&0xf);
  printf("\tlocalOvfl      : %x\n",(_status>>24)&0xf);
  printf("\tremoteData     : %x\n",_remoteUserData);
  printf("\tcellErrors     : %x\n",_cellErrCount);
  printf("\tlinkDown       : %x\n",_linkDownCount);
  printf("\tlinkErrors     : %x\n",_linkErrCount);
  printf("\tremoteOvflVC   : %x %x %x %x\n",
         _remoteOvfVc0,_remoteOvfVc1,_remoteOvfVc2,_remoteOvfVc3);
  printf("\tframesRxErr    : %x\n",_rxFrameErrs);
  printf("\tframesRx       : %x\n",_rxFrames);
  printf("\tlocalOvflVC   : %x %x %x %x\n",
         _localOvfVc0,_localOvfVc1,_localOvfVc2,_localOvfVc3);
  printf("\tframesTxErr    : %x\n",_txFrameErrs);
  printf("\tframesTx       : %x\n",_txFrames);
  printf("\trxClkFreq      : %f MHz\n",double(_rxClkFreq)*1.e-6);
  printf("\ttxClkFreq      : %f MHz\n",double(_txClkFreq)*1.e-6);
  printf("\tlastTxOp       : %x\n",_lastTxOpcode);
  printf("\tlastRxOp       : %x\n",_lastRxOpcode);
  printf("\tnTxOps         : %x\n",_txOpcodes);
  printf("\tnRxOps         : %x\n",_rxOpcodes);
}
