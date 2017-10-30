#ifndef Pds_RingBuffer_hh
#define Pds_RingBuffer_hh

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
      uint32_t   _csr;
      uint32_t   _dump[0x1fff];
    };
  };
};

#endif
