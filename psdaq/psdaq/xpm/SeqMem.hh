#ifndef Pds_XpmSeqMem_hh
#define Pds_XpmSeqMem_hh

#include "psdaq/cphw/Reg.hh"

namespace Pds {
  namespace Xpm {
    class SeqMem {
    public:
      Cphw::Reg& operator[](unsigned index) { return _word[index]; }
    public:
      Cphw::Reg _word[2048];
    };
  };
};

#endif
