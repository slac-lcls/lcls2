#ifndef HSD_HdrFifo_hh
#define HSD_HdrFifo_hh

#include <stdint.h>

namespace Pds {
  namespace HSD {
    class HdrFifo {
    public:
      uint32_t _wrFifoCnt;
      uint32_t _rdFifoCnt;
    };
  };
};

#endif
