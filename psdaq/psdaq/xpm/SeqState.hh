//
//  Builder for the SeqState application registers
//
#ifndef SeqState_hh
#define SeqState_hh

#include "psdaq/cphw/Reg.hh"

namespace Pds {
  namespace Xpm {
    class SeqState {
    public:
      Cphw::Reg countRequests;
      Cphw::Reg countInvalid;
      Cphw::Reg address;
      Cphw::Reg condcnt;
      const uint8_t* condCount() const { 
        uint32_t v = condcnt;
        return reinterpret_cast<const uint8_t*>(v);
      }
    };
  };
};

#endif
