#ifndef HSD_Module64_hh
#define HSD_Module64_hh

#include "psdaq/hsd/EnvMon.hh"
#include "psdaq/hsd/Globals.hh"
#include "psdaq/hsd/I2cSwitch.hh"
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
    class HdrFifo;
    class Pgp;

    class Module64 {
    public:
      //
      //  High level API
      //
      static Module64* create(int fd);
      static Module64* create(void*);

      ~Module64();

      uint64_t device_dna() const;

      void setup_timing(TimingType);
      void board_status();

      void set_local_id(unsigned bus);

      //  Initialize busses
      void init();

      //  Initialize clock tree and IO training
      void fmc_init          (TimingType =LCLS);
      void fmc_clksynth_setup(TimingType =LCLS);
      void fmc_dump();
      void fmc_modify(int,int,int,int,int,int);

      enum TestPattern { Ramp=0 };
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
      TprCore&    tpr    ();

      void setRxAlignTarget(unsigned);
      void setRxResetLength(unsigned);
      void dumpRxAlign     () const;
      void dumpPgp         () const;
      void dumpBase        () const;
      void dumpMap         () const;

      FexCfg*   fex     ();
      HdrFifo*  hdrFifo ();
      uint32_t* trgPhase();

      //  Zero copy read semantics
      //      ssize_t dequeue(void*&);
      //      void    enqueue(void*);
      //  Copy on read
      int read(uint32_t* data, unsigned data_size);

      void* reg();
      std::vector<Pgp*> pgp();
      Pds::Mmhw::Jtag*  xvc();

      //  Monitoring
      void   mon_start();
      EnvMon mon() const;

      void i2c_lock  (I2cSwitch::Port);
      void i2c_unlock();
    private:
      Module64() : _sem_i2c(Semaphore::FULL) {}

      class PrivateData;
      PrivateData* p;

      int       _fd;
      mutable Semaphore _sem_i2c;
    };
  };
};

#endif
