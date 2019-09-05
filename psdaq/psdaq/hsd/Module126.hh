#ifndef HSD_Module126_hh
#define HSD_Module126_hh

#include "psdaq/hsd/EnvMon.hh"
#include "psdaq/hsd/Globals.hh"
#include "psdaq/hsd/I2cSwitch.hh"
#include "psdaq/hsd/I2c126.hh"
#include "psdaq/service/Semaphore.hh"
#include <stdint.h>
#include <stdio.h>
#include <vector>

namespace Pds {
  namespace Mmhw {
    class AxiVersion;
    class Jtag;
  };
  namespace HSD {
    class TprCore;
    class FexCfg;
    class FlashController;
    class HdrFifo;
    class PhaseMsmt;
    class Pgp;
    class OptFmc;

    class Module126 {
    public:
      //
      //  High level API
      //
      static Module126* create(int fd);

      ~Module126();

      uint64_t device_dna() const;

      void setup_timing();
      void board_status();

      void set_local_id(unsigned bus);
      unsigned remote_id() const;

      void flash_write(const char*);
      FlashController& flash();

      //  Initialize busses
      void init();

      //  Initialize clock tree and IO training
      void fmc_init          (TimingType =LCLS);
      void fmc_clksynth_setup(TimingType =LCLS);
      void fmc_dump();
      void fmc_modify(int,int,int,int,int,int);

      int  train_io(unsigned);

      enum TestPattern { Ramp=0, Flash11=1, Flash12=3, Flash16=5, DMA=8 };
      void enable_test_pattern(TestPattern);
      void disable_test_pattern();
      void clear_test_pattern_errors();

      void enable_cal ();
      void disable_cal();
      void setAdcMux(unsigned channels);
      void setAdcMux(bool     interleave,
                     unsigned channels);

      void sample_init (unsigned length,
                        unsigned delay,
                        unsigned prescale);

      void trig_lcls  (unsigned eventcode);
      void trig_lclsii(unsigned fixedrate);
      void trig_daq   (unsigned partition);
      void trig_shift (unsigned shift);

      void start      ();
      void stop       ();

      //  Calibration
      unsigned get_offset(unsigned channel);
      unsigned get_gain  (unsigned channel);
      void     set_offset(unsigned channel, unsigned value);
      void     set_gain  (unsigned channel, unsigned value);
      void     sync      ();
      void     clocktree_sync();

      const Pds::Mmhw::AxiVersion& version() const;
      I2c126&                      i2c()     const;
      Pds::HSD::TprCore&    tpr    ();

      void setRxAlignTarget(unsigned);
      void setRxResetLength(unsigned);
      void dumpRxAlign     () const;
      void dumpPgp         () const;
      void dumpBase        () const;
      void dumpMap         () const;

      //  Zero copy read semantics
      //      ssize_t dequeue(void*&);
      //      void    enqueue(void*);
      //  Copy on read
      int read(uint32_t* data, unsigned data_size);

      FexCfg*                     fex     ();
      HdrFifo*                    hdrFifo ();
      PhaseMsmt*                  trgPhase();

      std::vector<Pgp*>            pgp    ();
      Pds::Mmhw::Jtag*             xvc    ();
      OptFmc&                      optfmc ();
      void*                        reg    ();

      //  Monitoring
      void   mon_start();
      EnvMon mon() const;

      void   i2c_lock  (I2cSwitch::Port) const;
      void   i2c_unlock() const;
    private:
      Module126() : _sem_i2c(Semaphore::FULL) {}

      class PrivateData;
      PrivateData* p;

      int       _fd;
      mutable Semaphore _sem_i2c;
    };
  };
};

#endif
