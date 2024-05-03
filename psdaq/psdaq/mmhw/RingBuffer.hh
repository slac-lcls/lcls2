#ifndef Pds_RingBuffer_hh
#define Pds_RingBuffer_hh

#include "psdaq/mmhw/Reg.hh"

#include <stdint.h>

namespace Pds {
  namespace Mmhw {

    class RingBuffer {
    public:
      RingBuffer() {}
    public:
      void     enable (bool);
      void     clear  ();
      void     dump   ();
    private:
      Reg   _csr;
      Reg   _dump[0x1fff];
    };
  };
};

#endif
