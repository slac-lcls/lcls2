#ifndef Kcu_Si570_hh
#define Kcu_Si570_hh

#include "psdaq/mmhw/Reg.hh"

namespace Drp {
  class Si570 {
  public:
    Si570();
    ~Si570();
  public:
    void   reset();   // Back to factory defaults
    void   program(int index=1); // Set for 185.7 MHz
    double read();    // Read factory calibration
  private:
    Pds::Mmhw::Reg _reg[256];
  };
};

#endif
