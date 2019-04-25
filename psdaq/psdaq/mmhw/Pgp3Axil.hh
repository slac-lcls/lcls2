#ifndef Pds_Pgp3Axil_hh
#define Pds_Pgp3Axil_hh

#include <unistd.h>
#include <stdint.h>

namespace Pds {
  namespace Mmhw {
    class Pgp3AxilBase {
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
      uint32_t  rsvd_40[0x10];
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
      uint32_t  rsvd_B0[0x14];
    };
    class Pgp3Axil : public Pgp3AxilBase {
    public:
      uint32_t  reserved[0xF00>>2];
    };
  };
};

#endif
