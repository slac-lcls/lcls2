#include "psdaq/hsd/PV64Ctrls.hh"
#include "psdaq/hsd/Module64.hh"
#include "psdaq/hsd/FexCfg.hh"
#include "psdaq/hsd/Fmc134Cpld.hh"
#include "psdaq/hsd/Fmc134Ctrl.hh"
#include "psdaq/hsd/QABase.hh"
#include "psdaq/hsd/Pgp.hh"
#include "psdaq/hsd/Jesd204b.hh"
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

static std::string STOU(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c){ return std::toupper(c); }
                 );
  return s;
}

static bool _interleave = false;

namespace Pds {
  namespace HSD {

    class PVC_Routine : public Routine {
    public:
      PVC_Routine(PV64Ctrls& pvc, Action a) : _pvc(pvc), _a(a) {}
      void routine() {
        switch(_a) {
        case Configure  : _pvc.configure(); break;
        case Reset      : _pvc.reset(); break;
        default: break;
        }
      }
    private:
      PV64Ctrls& _pvc;
      Action   _a;
    };
      

#define Q(a,b)      a ## b
#define PV(name)    Q(name, PV)

#define CPV(name, updatedBody, connectedBody)                           \
                                                                        \
    class PV(name) : public EpicsPVA,                                    \
                     public PVMonitorCb                                 \
    {                                                                   \
    public:                                                             \
      PV(name)(PV64Ctrls& ctrl, const char* pvName) :                     \
        EpicsPVA(pvName, this),                                          \
        _ctrl(ctrl) {}                                                  \
      virtual ~PV(name)() {}                                            \
    public:                                                             \
      virtual void updated();                                                   \
      virtual void onConnect();                                             \
    public:                                                             \
      void put() { if (this->EpicsPVA::connected())  _channel.put(); }   \
    private:                                                            \
      PV64Ctrls& _ctrl;                                                   \
    };                                                                  \
    void PV(name)::updated()                                            \
    {                                                                   \
      updatedBody                                                       \
    }                                                                   \
    void PV(name)::onConnect()                                          \
    {                                                                   \
      connectedBody                                                     \
    }
    
    CPV(ApplyConfig ,{if (getScalarAs<unsigned>()) _ctrl.call(Configure  );}, {})
    CPV(State       ,{}, {})
    CPV(Reset       ,{if (getScalarAs<unsigned>()) _ctrl.call(Reset      );}, {})
    CPV(PgpLoopback ,{_ctrl.loopback (getScalarAs<unsigned>()!=0);   }, {})

    PV64Ctrls::PV64Ctrls(Module64& m, Pds::Task& t) : _pv(0), _m(m), _task(t) {}
    PV64Ctrls::~PV64Ctrls() {}

    void PV64Ctrls::call(Action a) { _task.call(new PVC_Routine(*this, a)); }

    void PV64Ctrls::allocate(const std::string& title)
    {
      std::ostringstream o;
      o << title << ":";
      std::string pvbase = o.str();

      _state_pv = new EpicsPVA((pvbase+"BASE:READY").c_str());
      _setState(InTransition);

      for(unsigned i=0; i<_pv.size(); i++)
        delete _pv[i];
      _pv.resize(0);
      
#define NPV(name,pv)  _pv.push_back( new PV(name)(*this, (pvbase+pv).c_str()) )
#define NPV1(name)  _pv.push_back( new PvServer(STOU(pvbase+#name).c_str()) )

      NPV1(Enable);
      NPV1(Raw_Start);
      NPV1(Raw_Gate);
      NPV1(Raw_PS);
      NPV1(Fex_Start);
      NPV1(Fex_Gate);
      NPV1(Fex_PS);
      NPV1(Fex_Ymin);
      NPV1(Fex_Ymax);
      NPV1(Fex_Xpre);
      NPV1(Fex_Xpost);
      NPV1(FullEvt);
      NPV1(FullSize);
      NPV1(TestPattern);
      NPV1(SyncELo);
      NPV1(SyncEHi);
      NPV1(SyncOLo);
      NPV1(SyncOHi);
      NPV1(TrigShift);
      NPV1(PgpSkpIntvl);
      _pv.push_back(new PvServer((pvbase+"BASE:INTTRIGVAL" ).c_str()));
      _pv.push_back(new PvServer((pvbase+"BASE:INTAFULLVAL").c_str()));
      _pv.push_back(new PvServer((pvbase+"BASE:PARTITION"  ).c_str()));
      
      NPV(ApplyConfig,"BASE:APPLYCONFIG");
      NPV(Reset      ,"RESET");
      NPV(PgpLoopback,"PGPLOOPBACK");

      _m.setAdcMux(0x3);  // enable header cache for both channels

      _setState(Unconfigured);
    }

    //  enumeration of PV insert order above
    enum PvIndex { Enable, 
                   Raw_Start, Raw_Gate, Raw_PS, 
                   Fex_Start, Fex_Gate, Fex_PS, 
                   Fex_Ymin, Fex_Ymax, Fex_Xpre, Fex_Xpost,
                   FullEvt, FullSize,
                   TestPattern, SyncELo, SyncEHi, SyncOLo, SyncOHi, 
                   TrigShift, PgpSkpIntvl,
                   IntTrigVal, IntAFullVal, Partition, LastPv };

    Module64& PV64Ctrls::module() { return _m; }

    void PV64Ctrls::configure() {
      _setState(Unconfigured);

      _m.stop();

      // Update all necessary PVs
      for(unsigned i=0; i<LastPv; i++) {
        PvServer* pv = static_cast<PvServer*>(_pv[i]);
        while (!pv->EpicsPVA::connected()) {
          printf("pv[%u] not connected\n",i);
          usleep(100000);
        }
        pv->update();
      }

      { unsigned i=0;
        while(i<FullEvt) {
          printf("pv[%u] :",i);
          pvd::shared_vector<const unsigned> vec;
          _pv[i]->getVectorAs<unsigned>(vec);
          for(unsigned j=0; j<vec.size(); j++ )
            printf(" %u", vec[j]);
          printf("\n");
          i++;
        }
        while(i<LastPv) {
          printf("pv[%u] : %u\n",i,_pv[i]->getScalarAs<unsigned>());
          i++;
        }
      }

      { unsigned v = _pv[PgpSkpIntvl]->getScalarAs<unsigned>();
        std::vector<Pgp*> pgp = _m.pgp();
        for(unsigned i=0; i<pgp.size(); i++)
          pgp[i]->skip_interval(v);
      }

      _m.dumpPgp();

      unsigned channelMask = 0;
      pvd::shared_vector<const unsigned> vec;
      _pv[Enable]->getVectorAs(vec);

      for(unsigned i=0; i<vec.size(); i++)
        if (vec[i])
          channelMask |= (1<<i);

      //      unsigned mirrorMask = channelMask ^ 3;
      unsigned mirrorMask = 0;
      printf("channelMask 0x%x: mirrorMask 0x%x: ilv %c\n",
             channelMask, mirrorMask, _interleave?'T':'F');

      _m.setAdcMux(channelMask|mirrorMask);  // enable header cache for active channels

      // set testpattern
      int pattern = _pv[TestPattern]->getScalarAs<int>();
      if (pattern != 4 && pattern !=5)
        pattern = 0;
      printf("pattern %d\n",pattern);

      _m.i2c_lock(I2cSwitch::PrimaryFmc);
      Fmc134Ctrl* ctrl = reinterpret_cast<Fmc134Ctrl*>((char*)_m.reg()+0x81000);
      Fmc134Cpld* cpld = reinterpret_cast<Fmc134Cpld*>((char*)_m.reg()+0x12800);
      /*
      cpld->default_clocktree_init();
      cpld->default_adc_init();
      uint32_t* jesd0  = reinterpret_cast<uint32_t*  >((char*)_m.reg()+0x9B000);
      uint32_t* jesd1  = reinterpret_cast<uint32_t*  >((char*)_m.reg()+0x9B800);
      jesd0[0] = 0xff;
      jesd1[0] = 0xff;
      jesd0[4] = 0x23;
      jesd1[4] = 0x23;
      */
      ctrl->default_init(*cpld, pattern);
      ctrl->dump();
      _m.i2c_unlock();

      QABase& base = *reinterpret_cast<QABase*>((char*)_m.reg()+0x80000);
      base.init();
      base.resetCounts();

      { std::vector<Pgp*> pgp = _m.pgp();
        for(unsigned i=0; i<pgp.size(); i++)
          pgp[i]->resetCounts(); }

      unsigned partition = _pv[Partition]->getScalarAs<unsigned>();
      _m.trig_daq(partition);

      // configure fex's for each channel
      unsigned fullEvt  = _pv[FullEvt ]->getScalarAs<unsigned>();
      unsigned fullSize = _pv[FullSize]->getScalarAs<unsigned>();
      unsigned sumStreamMask=0;
      for(unsigned i=0; i<4; i++) {
        FexCfg& fex = _m.fex()[i];
        bool mirror = (1<<i)&mirrorMask;
        if (((1<<i)&channelMask) || mirror) {
          unsigned j=i;
          if (mirror) 
            i=__builtin_ffs(channelMask)-1;
          unsigned streamMask=0;
          if (_pv[Raw_PS]->getVectorElemAt<unsigned>(i)) {
            streamMask |= (1<<0);
            fex._base[0].setGate(_pv[Raw_Start]->getVectorElemAt<unsigned>(i),
                                 _pv[Raw_Gate ]->getVectorElemAt<unsigned>(i));
            fex._base[0].setFull(fullSize,fullEvt);
            fex._base[0]._prescale=_pv[Raw_PS]->getVectorElemAt<unsigned>(i)-1;
          }
          if (_pv[Fex_PS]->getVectorElemAt<unsigned>(i)) {
            streamMask |= (1<<1);
            fex._base[1].setGate(_pv[Fex_Start]->getVectorElemAt<unsigned>(i),
                                 _pv[Fex_Gate ]->getVectorElemAt<unsigned>(i));
            fex._base[1].setFull(fullSize,fullEvt);
            fex._base[1]._prescale=_pv[Fex_PS]->getVectorElemAt<unsigned>(i)-1;
            fex._stream[1].parms[0].v=_pv[Fex_Ymin]->getVectorElemAt<unsigned>(i);
            fex._stream[1].parms[1].v=_pv[Fex_Ymax]->getVectorElemAt<unsigned>(i);
            fex._stream[1].parms[2].v=_pv[Fex_Xpre]->getVectorElemAt<unsigned>(i);
            fex._stream[1].parms[3].v=_pv[Fex_Xpost]->getVectorElemAt<unsigned>(i);
          }
          fex._streams= streamMask;
          sumStreamMask |= streamMask;
          if (mirror) 
            i=j;
        }
        else
          fex._streams= 0;
      }


#define PRINT_FEX_FIELD(title,arg,op) {                                 \
        printf("%12.12s:",title);                                       \
        for(unsigned i=0,mask=channelMask; mask; i++,mask>>=1) {        \
          if ((mask&1)==0) continue;                                    \
          for(unsigned j=0; sumStreamMask>>j; j++)                      \
            printf("%c%u",                                              \
                   j==0?' ':'/',                                        \
                   fex[i]._base[j].arg op);                             \
        }                                                               \
        printf("\n"); }                             

      const FexCfg* fex = _m.fex();
      if (sumStreamMask) {
        PRINT_FEX_FIELD("GateBeg", _gate, &0x3fff);
        PRINT_FEX_FIELD("GateLen", _gate, >>16&0x3fff);
        PRINT_FEX_FIELD("FullRow", _full, &0xffff);
        PRINT_FEX_FIELD("FullEvt", _full, >>16&0x1f);
        PRINT_FEX_FIELD("Prescal", _prescale, &0x3ff);
      }

      printf("streams:");
      for(unsigned i=0,mask=(channelMask|mirrorMask); mask; i++,mask>>=1) {
        if ((mask&1)==0) continue;
        printf(" %2u", fex[i]._streams &0xf);
      }
      printf("\n");

#undef PRINT_FEX_FIELD

      base.dump();

      printf("Configure done\n");

      _m.start();
      _setState(Configured);
    }

    void PV64Ctrls::reset() {
      QABase& base = *reinterpret_cast<QABase*>((char*)_m.reg()+0x80000);
      base.resetFbPLL();
      usleep(1000000);
      base.resetFb ();
      base.resetDma();
      usleep(1000000);
    }

    void PV64Ctrls::loopback(bool v) {
      std::vector<Pgp*> pgp = _m.pgp();
      for(unsigned i=0; i<pgp.size(); i++)
        pgp[i]->loopback(v);
    }

    void PV64Ctrls::interleave(bool v) { _interleave = v; }

    void PV64Ctrls::_setState(State a) {
      unsigned v(a);
      _state_pv->putFrom(v);
    }
  };
};
