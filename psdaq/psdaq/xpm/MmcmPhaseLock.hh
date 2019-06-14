#ifndef Pds_Xpm_MmcmPhaseLock_hh
#define Pds_Xpm_MmcmPhaseLock_hh

#include "psdaq/cphw/Reg.hh"

namespace Pds {
  namespace Xpm {
    class MmcmPhaseLock {
    public:
      bool ready() const { return (delayValue & (1<<30))==0; }
      void reset() { _reset = 1; }
    public:
      Cphw::Reg delaySet;
      Cphw::Reg delayValue;
      Cphw::Reg ramAddr;
      Cphw::Reg ramData;
    private:
      Cphw::Reg _reset;
      uint32_t rsvd[(0x00100000-20)>>2];
    };
  };
};

#endif
