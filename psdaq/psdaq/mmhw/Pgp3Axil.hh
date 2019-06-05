#ifndef Pds_Pgp3Axil_hh
#define Pds_Pgp3Axil_hh

#include <unistd.h>
#include <stdint.h>

namespace Pds {
  namespace Mmhw {
    class Pgp3Axil {
    public:
      uint32_t  countReset;
      uint32_t  autoStatus;
      uint32_t  loopback;
      uint32_t  skpInterval;
      uint32_t  rxStatus; // phyRxActive, locLinkReady, remLinkReady
      uint32_t  cellErrCnt;
      uint32_t  linkDownCnt;
      uint32_t  linkErrCnt;
      uint32_t  remRxOflow; // +pause
      uint32_t  rxFrameCnt;
      uint32_t  rxFrameErrCnt;
      uint32_t  rxClkFreq;
      uint32_t  rxOpCodeCnt;
      uint32_t  rxOpCodeLast;
      uint32_t  rxOpCodeNum;
      uint32_t  rsvd_3C;
      uint32_t  rsvd_40[0x40>>2];
      // tx
      uint32_t  cntrl; // flowCntDis, txDisable
      uint32_t  txStatus; // phyTxActive, linkReady
      uint32_t  rsvd_88;
      uint32_t  locStatus; // locOflow, locPause
      uint32_t  txFrameCnt;
      uint32_t  txFrameErrCnt;
      uint32_t  rsvd_98;
      uint32_t  txClkFreq;
      uint32_t  txOpCodeCnt;
      uint32_t  txOpCodeLast;
      uint32_t  txOpCodeNum;
      uint32_t  rsvd_AC;
      uint32_t  rsvd_B0[0x50>>2];
      // phy
      uint64_t  phyData;
      uint32_t  phyHeader;
      uint32_t  rsvd_10C;
      uint64_t  ebData;
      uint32_t  ebHeader;
      uint32_t  ebOflow;
      uint32_t  gearboxAligned;
      uint32_t  rsvd_124[0xC>>2];
      uint32_t  rxInitCnt;
      uint32_t  rsvd_134[(0x1000-0x134)>>2];
    };
  };
};

#endif
