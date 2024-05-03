#ifndef HSD_HdrFifo_hh
#define HSD_HdrFifo_hh

#include "psdaq/mmhw/Reg.hh"
#include "Globals.hh"

namespace Pds {
  namespace HSD {
    class HdrFifo {
    public:
      Mmhw::Reg _wrFifoCnt;
      Mmhw::Reg _rdFifoCnt;
    };
  };
};

#endif
