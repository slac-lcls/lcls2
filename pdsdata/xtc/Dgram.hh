#ifndef Pds_Dgram_hh
#define Pds_Dgram_hh

#include "Sequence.hh"
#include "Env.hh"
#include "Xtc.hh"

namespace Pds {

#define PDS_DGRAM_STRUCT Sequence seq; Env env; Xtc xtc

  class Dgram {
  public:
    PDS_DGRAM_STRUCT;
  };

}
#endif
