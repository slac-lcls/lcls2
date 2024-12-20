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

#include "psdaq/mmhw/TprCore.hh"
#include "psdaq/mmhw/TriggerEventManager2.hh"
#include "psdaq/epicstools/EpicsPVA.hh"

#include <algorithm>
#include <sstream>
#include <string>
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

      unsigned group = PVGET(readoutGroup);
      _m.tem().det(fmc).stop();

      ChipAdcReg& reg = _m.chip(fmc).reg;
      reg.stop();

      _m.dumpPgp();

      // Let any configure set the group, so messageDelay will be updated
      _m.tem().det(fmc).group = group;

      _m.i2c_lock(I2cSwitch::PrimaryFmc);

      int testp = PVGET(test_pattern);
      if (testp!=int(_testpattern)) {
        if (testp>=0)
          _m.enable_test_pattern(Module134::TestPattern::Transport);
        else
          _m.disable_test_pattern();
        _testpattern = testp;
      }

      _m.i2c().fmc_cpld.adc_input(fmc,PVGET(input_chan));
      _m.i2c().fmc_cpld.adc_range(fmc,PVGET(fs_range_vpp));

      _m.i2c_unlock();
      
      FexCfg& fex = _m.chip(fmc).fex;

      if (PVGET(enable)==1) {

        _m.jesd(fmc).clearErrors();

        reg.init();
        reg.resetCounts();

        { std::vector<Pgp*> pgp = _m.pgp();
          for(unsigned i=4*fmc; i<4*(fmc+1); i++)
            pgp[i]->resetCounts(); }

        _configure_fex(fmc,fex);

        printf("Configure done for chip %d\n",fmc);

        reg.setChannels(1);
        reg.start();

        _m.tem().det(fmc).start(group);
      }
      else {
        fex.disable();
      }

      _ready[fmc]->putFrom<unsigned>(1);
    }

    void PV134Ctrls::configPgp(unsigned fmc) {
      Pds_Epics::EpicsPVA& pv = *_pv[4+fmc];
      std::vector<Pgp*> pgp = _m.pgp();

#define PVGETV(field) {                                 \
          pvd::shared_vector<const int> v;              \
          pv.getVectorAs(v,#field);                     \
          for(unsigned i=4*fmc; i<4*(fmc+1); i++)       \
              pgp[i]->tx##field(v[i]);                  \
      }
      
      PVGETV(diffctrl);
      PVGETV(precursor);
      PVGETV(postcursor);
    }

    void PV134Ctrls::reset(unsigned fmc) {
      Pds_Epics::EpicsPVA& pv = *_pv[2+fmc];
      if (PVGET(reset)) {
        { unsigned i=fmc;
          ChipAdcReg& reg = _m.chip(i).reg;
          reg.resetFbPLL();
          usleep(1000000);
          reg.resetFb ();
          reg.resetDma();
          usleep(1000000);
        }
      }
      if (PVGET(timpllrst)) {
        Pds::Mmhw::TprCore& tpr = _m.tpr();
        tpr.resetRxPll();
        usleep(10000);
        tpr.resetCounts();
      }
      if (PVGET(timrxrst)) {
        Pds::Mmhw::TprCore& tpr = _m.tpr();
        tpr.resetRx();
        usleep(10000);
        tpr.resetCounts();
      }
      loopback   (fmc,PVGET(pgploopback)!=0);
      disablefull(fmc,PVGET(pgpnofull)!=0);
      if (PVGET(jesdclear)) {
        printf("--jesdclear\n");
        for(unsigned j=0; j<8; j++) 
          _m.jesd(j).clearErrors();
      }
      if (PVGET(jesdsetup)) {
        printf("--jesdsetup\n");
        std::string adc[2];
        _m.setup_jesd(false,adc[0],adc[1]);
      }
      if (PVGET(jesdinit)) {
        printf("--jesdinit\n");
        _m.i2c_lock(I2cSwitch::PrimaryFmc);
        _m.jesdctl().default_init(_m.i2c().fmc_cpld,0);
        _m.i2c_unlock();
      }
      if (PVGET(jesdadcinit)) {
        _m.i2c_lock(I2cSwitch::PrimaryFmc);
        printf("--jesdadcinit\n");
        _m.jesdctl().reset();
        _m.i2c_unlock();
      }
      if (PVGET(cfgdump)) {
          for(unsigned i=0; i<2; i++) {
              FexCfg& fex = _m.chip(i).fex;
              unsigned streamMask = unsigned(fex._streams)&0xff;
#define PRINT_FEX_FIELD(title,arg,op) {                 \
                  printf("%12.12s:",title);             \
                  for(unsigned j=0; streamMask>>j; j++) \
                      printf("%c%u",                    \
                             j==0?' ':'/',              \
                             fex._base[j].arg op);      \
              }                                         \
              printf("\n");                              

              if (true) {
                  PRINT_FEX_FIELD("GateBeg", _reg[0], &0xffffffff);
                  PRINT_FEX_FIELD("GateLen", _reg[1], &0xfffff);
                  PRINT_FEX_FIELD("FullRow", _reg[2], &0xffff);
                  PRINT_FEX_FIELD("FullEvt", _reg[2], >>16&0x1f);
                  PRINT_FEX_FIELD("Prescal", _reg[1], >>20&0x3ff);
              }
#undef PRINT_FEX_FIELD

              printf("streams: %08x\n", unsigned(fex._streams));
          }
      }
    }
    void PV134Ctrls::loopback(unsigned fmc, bool v) {
      std::vector<Pgp*> pgp = _m.pgp();
      for(unsigned i=4*fmc; i<4*(fmc+1); i++)
        pgp[i]->loopback(v);

      usleep(1000);

      for(unsigned i=4*fmc; i<4*(fmc+1); i++)
        pgp[i]->resetCounts();
    }
    void PV134Ctrls::disablefull(unsigned fmc, bool v) {
      unsigned r = _m.optfmc().qsfp;
      if (v)
        r |= 0x100<<fmc;
      else
        r &= ~(0x100<<fmc);
      _m.optfmc().qsfp = r;
    }
  };
};
