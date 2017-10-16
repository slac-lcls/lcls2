#ifndef Xpm_RingBuffer_hh
#define Xpm_RingBuffer_hh

#include "psdaq/cphw/Reg.hh"

namespace Pds {
  namespace Cphw {
    class RingBuffer {
    public:
      RingBuffer() {}
    public:
      void     enable (bool);
      void     clear  ();
      void     dump   ();
    private:
      Reg   _csr;
      Reg   _dump[0x3ff];
    };
  };
};

#endif

