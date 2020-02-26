#ifndef Pds_Mmhw_Xvc_hh
#define Pds_Mmhw_Xvc_hh

#include <stdint.h>

namespace Pds {
  namespace Mmhw {

    class Jtag {
    public:
      uint32_t  length_offset;
      uint32_t  tms_offset;
      uint32_t  tdi_offset;
      uint32_t  tdo_offset;
      uint32_t  ctrl_offset;
    };

    class Xvc {
    public:
      static void* launch(Jtag*,
                          unsigned short port=2542,
                          bool           lverbose=false);
    };
  };
};

#endif
