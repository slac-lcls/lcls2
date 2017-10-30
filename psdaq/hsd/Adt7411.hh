#ifndef HSD_Adt7411_hh
#define HSD_Adt7411_hh

#include <stdint.h>

namespace Pds {
  namespace HSD {
    class Adt7411 {
    public:
      unsigned deviceId       () const;
      unsigned manufacturerId () const;
      unsigned interruptStatus() const;
      unsigned interruptMask  () const;
      unsigned internalTemp   () const;
      unsigned externalTemp   () const;
      void     dump           ();
    public:
      void     interruptMask  (unsigned);
    private:
      uint32_t _reg[256];
    };
  };
};

#endif
