#ifndef Pds_Pgp3Axil_hh
#define Pds_Pgp3Axil_hh

#include "Reg.hh"
#include "Reg64.hh"
#include <unistd.h>
#include <stdint.h>

namespace Pds {
  namespace Mmhw {
    class Pgp3Axil {
    public:
      Reg  countReset;
      Reg  autoStatus;
      Reg  loopback;
      Reg  skpInterval;
      Reg  rxStatus; // phyRxActive, locLinkReady, remLinkReady
      Reg  cellErrCnt;
      Reg  linkDownCnt;
      Reg  linkErrCnt;
      Reg  remRxOflow; // +pause
      Reg  rxFrameCnt;
      Reg  rxFrameErrCnt;
      Reg  rxClkFreq;
      Reg  rxOpCodeCnt;
      Reg  rxOpCodeLast;
      Reg  rxOpCodeNum;
      Reg  rsvd_3C;
      Reg  rsvd_40[0x40>>2];
      // tx
      Reg  cntrl; // flowCntDis, txDisable
      Reg  txStatus; // phyTxActive, linkReady
      Reg  rsvd_88;
      Reg  locStatus; // locOflow, locPause
      Reg  txFrameCnt;
      Reg  txFrameErrCnt;
      Reg  rsvd_98;
      Reg  txClkFreq;
      Reg  txOpCodeCnt;
      Reg  txOpCodeLast;
      Reg  txOpCodeNum;
      Reg  rsvd_AC;
      Reg  rsvd_B0[0x50>>2];
      // phy
      Reg64  phyData;
      Reg  phyHeader;
      Reg  rsvd_10C;
      Reg64  ebData;
      Reg  ebHeader;
      Reg  ebOflow;
      Reg  gearboxAligned;
      Reg  rsvd_124[0xC>>2];
      Reg  rxInitCnt;
      Reg  rsvd_134[(0x1000-0x134)>>2];
    };
  };
};

#endif
