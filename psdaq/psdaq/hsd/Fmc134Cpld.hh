#ifndef Fmc134CPld_hh
#define Fmc134CPld_hh

#include <stdint.h>

namespace Pds {
  namespace HSD {
    class Fmc134Cpld {
    public:
      Fmc134Cpld() {}
    public:
      void enableClock(bool);
      void initialize(bool lDualChannel=false,
                      bool lInternalRef=false);
      void enable_adc_prbs(bool);
      void enable_mon     (bool);
    public:
      void lmk_dump();
      void lmx_dump();
      void adc_dump(unsigned);
    private:
      void _hmc_init();
      void _lmk_init();
      void _lmx_init(bool);
      void _adc_init(unsigned,bool);
    public:
      enum DevSel { ADC0, ADC1, LMX, LMK, HMC };
      void     writeRegister( DevSel   dev,
                              unsigned address,
                              unsigned data );
      unsigned readRegister ( DevSel   dev,
                              unsigned address );
    private:
      unsigned _read();
    public:
      void dump() const;
    private:
      volatile uint32_t _command; // device select
      volatile uint32_t _control0; //
      volatile uint32_t _control1; //
      uint32_t _reserved3;
      volatile uint32_t _status;
      volatile uint32_t _version;
      volatile uint32_t _i2c_data[4]; // write cache
      volatile uint32_t _i2c_read[4]; // read cache
      uint32_t _reserved[0x100-14];
    };
  };
};

#endif
