#ifndef Pds_Cphw_GthRxAlign_hh
#define Pds_Cphw_GthRxAlign_hh

#include "psdaq/cphw/Reg.hh"

namespace Pds {
  namespace Cphw {
    class GthRxAlign {
    public:
      void setRxAlignTarget(unsigned);
      void setRxResetLength(unsigned);
      void dumpRxAlign     () const;
    public:
      Reg gthAlign[64];
      Reg gthAlignTarget;
      Reg gthAlignLast;
    };
  };
};

#endif

