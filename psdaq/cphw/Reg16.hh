#ifndef Cphw_Reg_hh
#define Cphw_Reg_hh

#include <stdint.h>

namespace Pds {
  namespace Cphw {
    class Reg {
    public:
      Reg& operator=(const unsigned);
      operator unsigned() const;
    public:
      void setBit  (unsigned);
      void clearBit(unsigned);
    public:
      static void set(const char* ip,
                      unsigned short port,
                      unsigned mem, 
                      unsigned long long memsz=(1ULL<<32));
    private:
      uint32_t _reserved;
    };
  };
};

#endif
