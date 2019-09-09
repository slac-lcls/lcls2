#ifndef HSD_HdrFifo_hh
#define HSD_HdrFifo_hh

#include "Globals.hh"

namespace Pds {
  namespace HSD {
    class HdrFifo {
    public:
      vuint32_t _wrFifoCnt;
      vuint32_t _rdFifoCnt;
    };
  };
};

#endif
