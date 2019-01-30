#ifndef Pds_Xpm_MmcmPhaseLock_hh
#define Pds_Xpm_MmcmPhaseLock_hh

#include "psdaq/cphw/Reg.hh"

namespace Pds {
  namespace Xpm {
    class MmcmPhaseLock {
    public:
      Cphw::Reg delaySet;
      Cphw::Reg delayValue;
      Cphw::Reg ramAddr;
      Cphw::Reg ramData;
    private:
      uint32_t rsvd[(0x00100000-16)>>2];
    };
  };
};

#endif
