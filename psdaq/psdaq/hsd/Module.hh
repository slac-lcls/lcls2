#ifndef HSD_Module_hh
#define HSD_Module_hh

#include "psdaq/hsd/Globals.hh"
#include <stdint.h>
#include <stdio.h>
#include <vector>

namespace Pds {
  namespace Mmhw {
    class AxiVersion;
  };
  namespace HSD {
    class TprCore;
    class FexCfg;
    class HdrFifo;
    class Pgp;

    class EnvMon {
    public:
      double local12v;
      double edge12v;
      double aux12v;
      double fmc12v;
      double local3_3v;
      double local2_5v;
      double local1_8v;
      double totalPower;
      double fmcPower;
      double boardTemp;
    };

    class Module {
    public:
      //
      //  High level API
      //
      static Module* create(int fd);
      static Module* create(int fd, TimingType);

      ~Module();

      uint64_t device_dna() const;

      void board_status();

      void flash_write(FILE*);

      //  Initialize busses
      void init();

      //  Initialize clock tree and IO training
      void fmc_init          (TimingType =LCLS);
      void fmc_clksynth_setup(TimingType =LCLS);
      void fmc_dump();

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

      const Pds::Mmhw::AxiVersion& version() const;
      Pds::HSD::TprCore&    tpr    ();

      void setRxAlignTarget(unsigned);
      void setRxResetLength(unsigned);
      void dumpRxAlign     () const;
      void dumpPgp         () const;
      void dumpBase        () const;
      void dumpMap         () const;

      FexCfg* fex();
      HdrFifo* hdrFifo();

      //  Zero copy read semantics
      //      ssize_t dequeue(void*&);
      //      void    enqueue(void*);
      //  Copy on read
      int read(uint32_t* data, unsigned data_size);

      void* reg();
      std::vector<Pgp*> pgp();

      //  Monitoring
      void   mon_start();
      EnvMon mon() const;
      
    private:
      Module() {}

      class PrivateData;
      PrivateData* p;

      int _fd;
    };
  };
};

#endif
