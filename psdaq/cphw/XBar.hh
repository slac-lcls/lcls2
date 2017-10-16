#ifndef Pds_Cphw_XBar_hh
#define Pds_Cphw_XBar_hh

#include "psdaq/cphw/Reg.hh"

namespace Pds {
  namespace Cphw {
    class XBar {
    public:
      enum Map { RTM0, FPGA, BP, RTM1 };
      void setOut( Map out, Map in );
      void dump  () const;
    public:
      Reg outMap[4];
    };
  }
}

#endif
