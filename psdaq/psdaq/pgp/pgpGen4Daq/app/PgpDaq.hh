#ifndef PgpDaq_hh
#define PgpDaq_hh

#include <stdint.h>

#define CLIENTS 4
#define LANES 4

namespace PgpDaq {

  class Client {
  public:
    uint32_t descAddrLo;
    uint32_t descAddrHi;
    uint32_t descFifoLo;
    uint32_t descFifoHi;
    uint32_t fifoDepth;
    uint32_t readIndex;
    uint32_t autoFill;
    uint32_t rsvd_1c;
  };

  class DmaLane {
  public:
    uint32_t client;
    uint32_t blockSize;
    uint32_t blocksPause;
    uint32_t rsvd_c;
    uint32_t fifoDepth;
    uint32_t memStatus;
    uint32_t queueCount;
    uint32_t ddrReadCmd;
  };

  class PgpLaneMisc {
  public:
    uint32_t loopback;
    uint32_t countReset;
    uint32_t dropCount;
    uint32_t truncCount;
    uint32_t rsvd_10[0x400-4];
  };

  class AxiStreamMonAxiL {
    uint32_t rsvd[0x400];
  };

  class PgpAxiL {
  public:
    uint32_t countReset;
    uint32_t autoStatus;
    uint32_t loopback;    // not implemented
    uint32_t skpInterval; // not implemented
    uint32_t rxStatus; // phyRxActive, locLinkReady, remLinkReady
    uint32_t cellErrCnt;
    uint32_t linkDwnCnt;
    uint32_t linkErrCnt;
    uint32_t remRxStatus; // remRxOflow, remRxPause
    uint32_t frameCount;
    uint32_t frameErrCnt;
    uint32_t rxClkFreq;
    uint32_t rxOpCodeCnt;
    uint32_t rxOpCodeData[2];
    uint32_t rsvd_803c;
    uint32_t remRxOflowCnt[16];
    uint32_t txControl; // flowCntlDis, txDisable
    uint32_t txStatus; // phyTxActive, txLinkReady
    uint32_t rsvd_8088;
    uint32_t locOflow;
    uint32_t txFrameCnt;
    uint32_t txFrameErrCnt;
    uint32_t rsvd_8098;
    uint32_t txClkFreq;
    uint32_t txOpCodeCnt;
    uint32_t txOpCodeData[2];
    uint32_t rsvd_80ac;
    uint32_t locTxOflowCnt[16];
    uint32_t rsvd_80f0[4];
    uint64_t rxPhyData;
    uint32_t rxPhyHeader;
    uint32_t rsvd_810c;
    uint64_t ebData;
    uint32_t ebHeader;
    uint32_t ebOflow;
    uint32_t gearboxStatus;
    uint32_t rsvd_8124[3];
    uint32_t phyRxInitCnt;
    uint32_t rsvd_8134[(0x1000-0x134)>>2];
  };

  class PgpLane {
  public:
    PgpLaneMisc      misc;
    AxiStreamMonAxiL monrx;
    AxiStreamMonAxiL montx;
    uint32_t         rsvd_3000[0x5000>>2];
    PgpAxiL          axil;
    //  Gthe3Channel (DRP)
    uint32_t rsvd_9000[0x400];
    //
    uint32_t rsvd_a000[0x1800];
  };

  class PgpTxSim {
  public:
    uint32_t overflow;
    uint32_t rsvd_4[0x40-1];
    uint32_t control;
    uint32_t size;
    uint32_t rsvd_48[0x00040000-0x42];
  };

  class PgpCard {
  public:
    unsigned nlanes  () const;
    unsigned nclients() const;
  public:
    uint32_t version;
    uint32_t scratch;
    uint32_t upTimeCnt;
    uint32_t rsvd_C[0x3d];
    uint32_t rsvd_100[0x1C0];
    uint32_t buildStr[64];

    uint32_t rsvd_00000900[0x00200000-0x240];
    uint32_t resources;          // @ 0x00800000
    uint32_t reset;
    uint32_t monSampleInterval;  // 200MHz counts
    uint32_t monReadoutInterval; // monSample counts
    uint32_t monEnable;
    uint32_t monHistAddrLo;
    uint32_t monHistAddrHi;
    uint32_t monSampleCounter;
    uint32_t monReadoutCounter;
    uint32_t monStatus;
    uint32_t rsvd_00800028[22];
    Client   clients[CLIENTS];   // @ 0x00800080
    DmaLane  dmaLane[LANES];     // @ 0x00800100
    uint32_t rsvd_00800180[0x00100000-0x60];
    PgpLane  pgpLane[LANES];     // @ 0x00C00000
    uint32_t rsvd_00C40000[0x00030000];
    PgpTxSim sim;                // @ 0x00D00000
    uint32_t rsvd_00e00000[0x00080000];
  };

  inline unsigned PgpCard::nlanes() const {
    return (this->resources>>0)&0xf;
  }

  inline unsigned PgpCard::nclients() const {
    return (this->resources>>4)&0xf;
  }
};

#endif
