#include "PV134Ctrls.hh"
#include "Module134.hh"
#include "FexCfg.hh"
#include "Fmc134Cpld.hh"
#include "Fmc134Ctrl.hh"
#include "I2c134.hh"
#include "ChipAdcCore.hh"
#include "ChipAdcReg.hh"
#include "Pgp.hh"
#include "OptFmc.hh"
#include "Jesd204b.hh"
#include "TprCore.hh"

#include "psdaq/mmhw/TriggerEventManager.hh"
#include "psdaq/epicstools/EpicsPVA.hh"

#include <algorithm>
#include <sstream>
#include <cctype>
#include <stdio.h>
#include <unistd.h>

using Pds_Epics::EpicsPVA;

using Pds::HSD::Jesd204b;
using Pds::HSD::Jesd204bStatus;

namespace Pds {
  namespace HSD {

    PV134Ctrls::PV134Ctrls(Module134& m, Pds::Task& t) :
      PVCtrlsBase(t),
      _m(m)
    {}

    void PV134Ctrls::_allocate()
    {
      _m.i2c_lock(I2cSwitch::PrimaryFmc);
      _m.jesdctl().default_init(_m.i2c().fmc_cpld, _testpattern=0);
      _m.jesdctl().dump();
      _m.i2c_unlock();

      _m.optfmc().qsfp = 0x89;
    }

    void PV134Ctrls::configure(unsigned fmc) {

#define PVGET(name) pv.getScalarAs<unsigned>(#name)
#define PVGETF(name) pv.getScalarAs<float>(#name)

      Pds_Epics::EpicsPVA& pv = *_pv[fmc];

      ChipAdcReg& reg = _m.chip(fmc).reg;
      reg.stop();

      _m.dumpPgp();

      _m.i2c_lock(I2cSwitch::PrimaryFmc);

      int testp = PVGET(test_pattern);
      if (testp!=int(_testpattern)) {
        if (testp>=0)
          _m.enable_test_pattern(Module134::TestPattern::Transport);
        else
          _m.disable_test_pattern();
        _testpattern = testp;
      }

      _m.i2c().fmc_cpld.adc_range(fmc,PVGET(fs_range_vpp));
      _m.i2c_unlock();

      FexCfg& fex = _m.chip(fmc).fex;

      if (PVGET(enable)==1) {

        reg.init();
        reg.resetCounts();

        { std::vector<Pgp*> pgp = _m.pgp();
          for(unsigned i=4*fmc; i<4*(fmc+1); i++)
            pgp[i]->resetCounts(); }

        _configure_fex(fmc,fex);

        printf("Configure done for chip %d\n",fmc);

        reg.setChannels(1);
        reg.start();

        unsigned group = PVGET(readoutGroup);
        _m.tem().det(fmc).start(group);
      }

      _ready[fmc]->putFrom<unsigned>(1);
    }

    void PV134Ctrls::reset() {
      Pds_Epics::EpicsPVA& pv = *_pv[_pv.size()-1];
      if (PVGET(reset)) {
        for(unsigned i=0; i<2; i++) {
          ChipAdcReg& reg = _m.chip(i).reg;
          reg.resetFbPLL();
          usleep(1000000);
          reg.resetFb ();
          reg.resetDma();
          usleep(1000000);
        }
      }
      if (PVGET(timpllrst)) {
        Pds::HSD::TprCore& tpr = _m.tpr();
        tpr.resetRxPll();
        usleep(10000);
        tpr.resetCounts();
      }
      if (PVGET(timrxrst)) {
        Pds::HSD::TprCore& tpr = _m.tpr();
        tpr.resetRx();
        usleep(10000);
        tpr.resetCounts();
      }
      if (PVGET(jesdclear)) {
        for(unsigned j=0; j<8; j++) 
          _m.jesd(j).clearErrors();
      }
    }
    void PV134Ctrls::loopback(bool v) {
      std::vector<Pgp*> pgp = _m.pgp();
      for(unsigned i=0; i<pgp.size(); i++)
        pgp[i]->loopback(v);
    }
  };
};
