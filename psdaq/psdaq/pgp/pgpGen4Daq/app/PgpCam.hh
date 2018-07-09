#ifndef PgpCam_hh
#define PgpCam_hh

#include <stdint.h>

namespace PgpDaq {

  class PgpCam {
  public:
    unsigned nlanes  () const;
    unsigned nclients() const;
  public:
    uint32_t version;
    uint32_t scratch;
    uint32_t upTimeCnt;
    uint32_t rsvd_C[0x3d];
    uint32_t rsvd_100[0x180];
    uint32_t dna[4];
    uint32_t rsvd_710[60];
    uint32_t buildStr[64];
    uint32_t rsvd_00000900[0x00200000-0x240];
    PgpLane  pgpLane[8];     // @ 0x00800000
    PgpTxSim sim;            // @ 0x00880000
  };

  inline unsigned PgpCam::nlanes() const {
    return 8;
  }

  inline unsigned PgpCam::nclients() const {
    return 0;
  }
};

#endif
