#ifndef PgpDaq_hh
#define PgpDaq_hh

#include <stdint.h>

#define CLIENTS 2
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
    uint32_t rsvd_18[2];
  };

  class PgpLane {
  public:
    uint32_t loopback;
    uint32_t rsvd_4[0x00004000-1];
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
    uint32_t rsvd_00800028[6];
    Client   clients[CLIENTS];
    DmaLane  dmaLane[LANES];
    uint32_t rsvd_00800080[0x00100000-0x40];
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
