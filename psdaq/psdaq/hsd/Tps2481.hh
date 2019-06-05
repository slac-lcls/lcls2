#ifndef HSD_Tsp2481_hh
#define HSD_Tsp2481_hh

#include "psdaq/mmhw/RegProxy.hh"
#include <stdint.h>

namespace Pds {
  namespace HSD {
    class Tps2481 {
    public:
      double   current_A      () const;
      double   power_W        () const;
      void     start          ();
      void     dump           ();
    private:
      Pds::Mmhw::RegProxy _cfg;
      Pds::Mmhw::RegProxy _shtv;
      Pds::Mmhw::RegProxy _busv;
      Pds::Mmhw::RegProxy _pwr;
      Pds::Mmhw::RegProxy _cur;
      Pds::Mmhw::RegProxy _cal;
      Pds::Mmhw::RegProxy _reg[250];
    };
  };
};

#endif
