#ifndef HSD_FmcSpi_hh
#define HSD_FmcSpi_hh

#include "psdaq/hsd/Globals.hh"
#include <stdint.h>

namespace Pds {
  namespace HSD {
    class FmcSpi {
    public:
      int initSPI();
      int resetSPIclocktree();
      int resetSPIadc      ();

      enum SyncSrc { NoSync=0, FPGA=1 };
      int cpld_init();
        
      int clocktree_init   (unsigned   clksrc, 
                            unsigned   vcotype,
                            TimingType timing = LCLS);
      void clockWhileSync  ();
      void limitBandwidth  (bool);
      int adc_enable_test  (unsigned pattern);
      int adc_disable_test ();

      void adc_enable_cal  ();
      void adc_disable_cal ();

      void setAdcMux(unsigned channels);
      void setAdcMux(bool     interleave,
                     unsigned channels);

      unsigned get_offset(unsigned channel);
      unsigned get_gain  (unsigned channel);
      void     set_offset(unsigned channel, unsigned value);
      void     set_gain  (unsigned channel, unsigned value);
    private:
      void     _writeAD9517(unsigned addr, unsigned value);
      void     _writeADC   (unsigned addr, unsigned value);
      void     _writeCPLD  (unsigned addr, unsigned value);
      unsigned _readAD9517 (unsigned addr);
      unsigned _readADC    (unsigned addr);
      unsigned _readCPLD   (unsigned addr);
      void     _applySync();
      char     _cardId() const;
    private:
      uint32_t _reg[256]; // Bridge configuration access
      uint32_t _sp1[256]; // SPI device access 1B address
      uint32_t _sp2[256]; // SPI device access 2B address
    };
  };
};

#endif
