#ifndef Pds_TriggerEventManager2_hh
#define Pds_TriggerEventManager2_hh

#include "psdaq/mmhw/TriggerEventManager.hh"

//
//  Modified layout from original TEM
//
#include <stdint.h>

namespace Pds {
  namespace Mmhw {
    class TriggerEventManager2 {
    public:
      TriggerEventManager2() {}
    public:
      XpmMessageAligner&  xma() { return _xma; }
      TriggerEventBuffer& det(unsigned i) { return reinterpret_cast<TriggerEventBuffer*>(this+1)[i]; }
    private:
      uint32_t          _rsvd_0[0x8000>>2];
      XpmMessageAligner _xma;
      uint32_t          _rsvd_1000[(0x1000-sizeof(_xma))>>2];
    };
  };
};

#endif
