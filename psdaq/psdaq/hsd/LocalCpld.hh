#ifndef HSD_LocalCpld_hh
#define HSD_LocalCpld_hh

#include <stdint.h>

namespace Pds {
  namespace HSD {
    class LocalCpld {
    public:
      unsigned revision  () const;
      unsigned GAaddr    () const;
    public:
      void     reloadFpga();
      void     GAaddr    (unsigned);
    private:
      uint32_t          _reg[256];
    };
  };
};

#endif
