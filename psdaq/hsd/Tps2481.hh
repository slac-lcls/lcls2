#ifndef HSD_Tsp2481_hh
#define HSD_Tsp2481_hh

#include <stdint.h>

namespace Pds {
  namespace HSD {
    class Tps2481 {
    public:
      void     dump           ();
    private:
      uint32_t _cfg;
      uint32_t _shtv;
      uint32_t _busv;
      uint32_t _pwr;
      uint32_t _cur;
      uint32_t _cal;
      uint32_t _reg[250];
    };
  };
};

#endif
