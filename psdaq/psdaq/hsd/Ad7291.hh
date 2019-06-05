#ifndef HSD_Ad7291_hh
#define HSD_Ad7291_hh

#include "psdaq/mmhw/RegProxy.hh"
#include <stdint.h>

namespace Pds {
  namespace HSD {
    class Ad7291_Mon {
    public:
      double Tint;  // degC
      double Vdd;   // Volt
      double ain[8];// Volt
    };

    class Ad7291 {
    public:
      void     start          ();
      Ad7291_Mon mon          ();
    private:
      unsigned _read(unsigned ch);
      //      uint32_t _reg[256];
      Pds::Mmhw::RegProxy _reg[256];
    };

    class FmcAdcMon {
    public:
      FmcAdcMon(Ad7291_Mon cur) : _mon(cur) {}
    public:
      double adc0_1p1v_ana() const { return _mon.ain[0]*0.00061035; }
      double adc0_1p1v_dig() const { return _mon.ain[1]*0.00061035; }
      double adc0_1p9v_ana() const { return _mon.ain[2]*0.00061035; }
      double adc1_1p1v_ana() const { return _mon.ain[3]*0.00061035; }
      double adc1_1p1v_dig() const { return _mon.ain[4]*0.00061035; }
      double adc1_1p9v_ana() const { return _mon.ain[5]*0.00061035; }
      double adc0_temp    () const { return 25.0+(0.75-_mon.ain[6]*0.00061035)/0.0016; }
      double adc1_temp    () const { return 25.0+(0.75-_mon.ain[7]*0.00061035)/0.0016; }
      double int_temp     () const { return _mon.Tint*0.25; }
    public:
      void dump() const;
    private:
      Ad7291_Mon _mon;
    };

    class FmcVMon {
    public:
      FmcVMon(const Ad7291_Mon& mon) : _mon(mon) {}
    public:
      double v5p5_osc100  () const { return _mon.ain[0]*0.00183105; }
      double v3p3_clock   () const { return _mon.ain[1]*0.0012207; }
      double v3p3_lmxpll  () const { return _mon.ain[2]*0.0012207; }
      double vp_cpld_1p8v () const { return _mon.ain[3]*0.00061035; }
      double vadj         () const { return _mon.ain[4]*0.0012207; }
      double fmc_3p3v     () const { return _mon.ain[5]*0.0012207; }
      double fmc_12p0v    () const { return _mon.ain[6]*0.0036621; }
      double vio_m2c      () const { return _mon.ain[7]*0.0012207; }
      double int_temp     () const { return _mon.Tint*0.25; }
    public:
      void dump() const;
    private:
      Ad7291_Mon _mon;
    };
  };
};

#endif
