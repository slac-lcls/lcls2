#ifndef Pds_Cphw_Xvc_hh
#define Pds_Cphw_Xvc_hh

#include "psdaq/cphw/Reg.hh"

namespace Pds {
  namespace Cphw {

    class Jtag {
    public:
      Reg  length_offset;
      Reg  tms_offset;
      Reg  tdi_offset;
      Reg  tdo_offset;
      Reg  ctrl_offset;
    };

    class Xvc {
    public:
      static void* launch(Jtag*,bool lverbose=false);
    };
  };
};

#endif
