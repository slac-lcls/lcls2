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

#include "psdaq/epicstools/EpicsPVA.hh"
#include "psdaq/epicstools/PvServer.hh"
#include "psdaq/service/Task.hh"
#include "psdaq/service/Routine.hh"

#include <algorithm>
#include <sstream>
#include <cctype>
#include <stdio.h>
#include <unistd.h>

using Pds_Epics::EpicsPVA;
using Pds_Epics::PvServer;
using Pds_Epics::PVMonitorCb;

using Pds::Routine;
using Pds::HSD::Jesd204b;
using Pds::HSD::Jesd204bStatus;

namespace Pds {
  namespace HSD {

    class PVC_Routine : public Routine {
    public:
      PVC_Routine(PV134Ctrls& pvc, Action a) : _pvc(pvc), _a(a) {}
      void routine() {
        switch(_a) {
        case ConfigureA : _pvc.configure(0); break;
        case ConfigureB : _pvc.configure(1); break;
        case Reset      : _pvc.reset     (); break;
        default: break;
        }
      }
    private:
      PV134Ctrls& _pvc;
      Action   _a;
    };
       
#define Q(a,b)      a ## b
#define PV(name)    Q(name, PV)

#define CPV(name, updatedBody, connectedBody)                           \
                                                                        \
    class PV(name) : public EpicsPVA,                                   \
                     public PVMonitorCb                                 \
    {                                                                   \
    public:                                                             \
      PV(name)(PV134Ctrls& ctrl, const char* pvName) :                  \
        EpicsPVA(pvName, this),                                         \
        _ctrl(ctrl) {}                                                  \
      virtual ~PV(name)() {}                                            \
    public:                                                             \
      virtual void updated();                                           \
      virtual void onConnect();                                         \
    public:                                                             \
      void put() { if (this->EpicsPVA::connected())  _channel.put(); }  \
    private:                                                            \
      PV134Ctrls& _ctrl;                                                \
    };                                                                  \
    void PV(name)::updated()                                            \
    {                                                                   \
      updatedBody                                                       \
        }                                                               \
    void PV(name)::onConnect()                                          \
    {                                                                   \
      connectedBody                                                     \
        }
    
    CPV(State       ,{}, {})
    CPV(Reset       ,{if (getScalarAs<unsigned>()) _ctrl.call(Reset      );}, {})
    CPV(PgpLoopback ,{_ctrl.loopback (getScalarAs<unsigned>()!=0);   }, {})
    CPV(ConfigA     ,{_ctrl.call(ConfigureA);}, {})
    CPV(ConfigB     ,{_ctrl.call(ConfigureB);}, {})

    PV134Ctrls::PV134Ctrls(Module134& m, Pds::Task& t) : _pv(0), _m(m), _task(t) 
    {}
    PV134Ctrls::~PV134Ctrls() {}

    void PV134Ctrls::call(Action a) { _task.call(new PVC_Routine(*this, a)); }

    void PV134Ctrls::allocate(const std::string& title)
    {
      std::ostringstream o;
      o << title << ":";
      std::string pvbase = o.str();

      _state_pv = new EpicsPVA((pvbase+"BASE:READY").c_str());
      _setState(InTransition);

      _readyA = new EpicsPVA((pvbase+"A:READY").c_str());
      _readyB = new EpicsPVA((pvbase+"B:READY").c_str());

      for(unsigned i=0; i<_pv.size(); i++)
        delete _pv[i];
      _pv.resize(0);
      
#define NPV(name,pv)  _pv.push_back( new PV(name)(*this, (pvbase+pv).c_str()) )

      NPV(ConfigA    ,"A:CONFIG");
      NPV(ConfigB    ,"B:CONFIG");

      _m.i2c_lock(I2cSwitch::PrimaryFmc);
      _m.jesdctl().default_init(_m.i2c().fmc_cpld, _testpattern=0);
      _m.jesdctl().dump();
      _m.i2c_unlock();

      _m.optfmc().qsfp = 0x89;

      _setState(Unconfigured);
    }

    //  enumeration of PV insert order above
    enum PvIndex { ConfigA, ConfigB, LastPv };

    Module134& PV134Ctrls::module() { return _m; }

    void PV134Ctrls::configure(unsigned fmc) {

#define PVGET(name) pv.getScalarAs<unsigned>(#name)
#define PVGETF(name) pv.getScalarAs<float>(#name)

      Pds_Epics::EpicsPVA& pv = *_pv[(fmc==0) ? ConfigA:ConfigB];

      ChipAdcReg& reg = _m.chip(fmc).reg;
      reg.stop();

      _m.i2c_lock(I2cSwitch::PrimaryFmc);

      unsigned testp = PVGET(test_pattern);
      if (testp!=_testpattern) {
        _m.jesdctl().default_init(_m.i2c().fmc_cpld, 
                                  _testpattern=testp);
        _m.jesdctl().dump();
      }

      _m.i2c().fmc_cpld.adc_range(fmc,PVGET(fs_range_vpp));
      _m.i2c_unlock();

      FexCfg& fex = _m.chip(fmc).fex;

      unsigned streamMask=0;
      if (PVGET(enable)==1) {

        reg.init();
        reg.resetCounts();

        { std::vector<Pgp*> pgp = _m.pgp();
          for(unsigned i=4*fmc; i<4*(fmc+1); i++)
            pgp[i]->resetCounts(); }

        unsigned group = PVGET(readoutGroup);
        reg.setupDaq(group);

        // configure fex's for each channel
        unsigned fullEvt  = PVGET(full_event);
        unsigned fullSize = PVGET(full_size);
        if (PVGET(raw_prescale)) {
          streamMask |= (1<<0);
          fex._base[0].setGate(PVGET(raw_start),
                               PVGET(raw_gate));
          fex._base[0].setFull(fullSize,fullEvt);
          fex._base[0]._prescale=PVGET(raw_prescale)-1;
        }
        if (PVGET(fex_prescale)) {
          streamMask |= (1<<1);
          fex._base[1].setGate(PVGET(fex_start),
                               PVGET(fex_gate));
          fex._base[1].setFull(fullSize,fullEvt);
          fex._base[1]._prescale=PVGET(fex_prescale)-1;
          fex._stream[1].parms[0].v=PVGET(fex_ymin);
          fex._stream[1].parms[1].v=PVGET(fex_ymax);
          fex._stream[1].parms[2].v=PVGET(fex_xpre);
          fex._stream[1].parms[3].v=PVGET(fex_xpost);
        }
        fex._streams= streamMask;
    
#define PRINT_FEX_FIELD(title,arg,op) {                                 \
          printf("%12.12s:",title);                                     \
          for(unsigned j=0; streamMask>>j; j++)                         \
            printf("%c%u",                                              \
                   j==0?' ':'/',                                        \
                   fex._base[j].arg op);                                \
        }                                                               \
        printf("\n");                              

        if (streamMask) {
          PRINT_FEX_FIELD("GateBeg", _gate, &0x3fff);
          PRINT_FEX_FIELD("GateLen", _gate, >>16&0x3fff);
          PRINT_FEX_FIELD("FullRow", _full, &0xffff);
          PRINT_FEX_FIELD("FullEvt", _full, >>16&0x1f);
          PRINT_FEX_FIELD("Prescal", _prescale, &0x3ff);
        }
#undef PRINT_FEX_FIELD

        printf("streams: %2u\n", fex._streams &0xf);

        printf("Configure done\n");

        reg.setChannels(1);
        reg.start();
      }

      if (fmc==0)
        _readyA->putFrom<unsigned>(1);
      else
        _readyB->putFrom<unsigned>(1);
    }

    void PV134Ctrls::reset() {
      for(unsigned i=0; i<2; i++) {
        ChipAdcReg& reg = _m.chip(i).reg;
        reg.resetFbPLL();
        usleep(1000000);
        reg.resetFb ();
        reg.resetDma();
        usleep(1000000);
      }
    }

    void PV134Ctrls::loopback(bool v) {
      std::vector<Pgp*> pgp = _m.pgp();
      for(unsigned i=0; i<pgp.size(); i++)
        pgp[i]->loopback(v);
    }

    void PV134Ctrls::_setState(State a) {
      unsigned v(a);
      _state_pv->putFrom(v);
    }
  };
};
