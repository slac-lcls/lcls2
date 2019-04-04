#include "psdaq/hsd/PVCtrls.hh"
#include "psdaq/hsd/Module.hh"
#include "psdaq/hsd/FexCfg.hh"
#include "psdaq/hsd/QABase.hh"
#include "psdaq/hsd/Pgp.hh"
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
      PVC_Routine(PVCtrls& pvc, Action a) : _pvc(pvc), _a(a) {}
      void routine() {
        switch(_a) {
        case Configure  : _pvc.configure(); break;
        case Reset      : _pvc.reset(); break;
        default: break;
        }
      }
    private:
      PVCtrls& _pvc;
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
      PV(name)(PVCtrls& ctrl, const char* pvName) :                     \
        EpicsPVA(pvName, this),                                          \
        _ctrl(ctrl) {}                                                  \
      virtual ~PV(name)() {}                                            \
    public:                                                             \
      virtual void updated();                                                   \
      virtual void onConnect();                                             \
    public:                                                             \
      void put() { if (this->EpicsPVA::connected())  _channel.put(); }   \
    private:                                                            \
      PVCtrls& _ctrl;                                                   \
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

    PVCtrls::PVCtrls(Module& m, Pds::Task& t) : _pv(0), _m(m), _task(t) {}
    PVCtrls::~PVCtrls() {}

    void PVCtrls::call(Action a) { _task.call(new PVC_Routine(*this, a)); }

    void PVCtrls::allocate(const std::string& title)
    {
      for(unsigned i=0; i<_pv.size(); i++)
        delete _pv[i];
      _pv.resize(0);
      
      std::ostringstream o;
      o << title << ":";
      std::string pvbase = o.str();

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

      _state_pv = new StatePV(*this, (pvbase+"BASE:READY").c_str());

      _setState(Unconfigure);
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

    Module& PVCtrls::module() { return _m; }

    void PVCtrls::configure() {
      _setState(Unconfigure);

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
        for(unsigned i=0; i<4; i++)
          pgp[i]->skip_interval(v);
      }

      _m.dumpPgp();

      // set trigger shift
      int shift = _pv[TrigShift]->getScalarAs<int>();
      _m.trig_shift(shift);

      // don't know how often this is necessary
      _m.train_io(0);

      unsigned channelMask = 0;
      for(unsigned i=0; i<4; i++) {
        if (_pv[Enable]->getVectorElemAt<unsigned>(i)) {
          channelMask |= (1<<i);
          if (_interleave) break;
        }
      }
      printf("channelMask 0x%x: ilv %c\n",
             channelMask, _interleave?'T':'F');
      _m.setAdcMux( _interleave, channelMask );

      // set testpattern
      int pattern = _pv[TestPattern]->getScalarAs<int>();
      printf("Pattern: %d\n",pattern);
      _m.disable_test_pattern();
      if (pattern>=0)
        _m.enable_test_pattern((Module::TestPattern)pattern);

      int ephlo = _pv[SyncELo]->getScalarAs<int>();
      int ephhi = _pv[SyncEHi]->getScalarAs<int>();
      int ophlo = _pv[SyncOLo]->getScalarAs<int>();
      int ophhi = _pv[SyncOHi]->getScalarAs<int>();
      while(1) {
        int eph = _m.trgPhase()[0];
        int oph = _m.trgPhase()[1];
        printf("trig phase %05d [%05d/%05d] %05d [%05d/%05d]\n",
               eph,ephlo,ephhi,
               oph,ophlo,ophhi);
        if (eph > ephlo && eph < ephhi &&
            oph > ophlo && oph < ophhi)
          break;
        _m.sync();
        usleep(100000);  // Wait for relock
      }

      // zero the testpattern error counts
      _m.clear_test_pattern_errors();

      // _m.sample_init(32+48*length, 0, 0);
      QABase& base = *reinterpret_cast<QABase*>((char*)_m.reg()+0x80000);
      base.init();
      base.resetCounts();

      { std::vector<Pgp*> pgp = _m.pgp();
        for(unsigned i=0; i<pgp.size(); i++)
          pgp[i]->resetCounts(); }

      //  These aren't used...
      //      base.samples = ;
      //      base.prescale = ;

      unsigned partition = _pv[Partition]->getScalarAs<unsigned>();
      _m.trig_daq(partition);

      // configure fex's for each channel
      unsigned fullEvt  = _pv[FullEvt ]->getScalarAs<unsigned>();
      unsigned fullSize = _pv[FullSize]->getScalarAs<unsigned>();
      for(unsigned i=0; i<4; i++) {
        FexCfg& fex = _m.fex()[i];
        if ((1<<i)&channelMask) {
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
        }
        else
          fex._streams= 0;
      }


#define PRINT_FEX_FIELD(title,arg,op) {                 \
        printf("%12.12s:",title);                       \
        for(unsigned i=0; i<4; i++) {                   \
          if (((1<<i)&channelMask)==0) continue;        \
          printf(" %u/%u/%u",                           \
                 fex[i]._base[0].arg op,                \
                 fex[i]._base[1].arg op,                \
                 fex[i]._base[2].arg op);               \
        }                                               \
        printf("\n"); }                             
  
      FexCfg* fex = _m.fex();
      PRINT_FEX_FIELD("GateBeg", _gate, &0x3fff);
      PRINT_FEX_FIELD("GateLen", _gate, >>16&0x3fff);
      PRINT_FEX_FIELD("FullRow", _full, &0xffff);
      PRINT_FEX_FIELD("FullEvt", _full, >>16&0x1f);
      PRINT_FEX_FIELD("Prescal", _prescale, &0x3ff);

      printf("streams:");
      for(unsigned i=0; i<4; i++) {
        if (((1<<i)&channelMask)==0) continue;
        printf(" %2u", fex[i]._streams &0xf);
      }
      printf("\n");

#undef PRINT_FEX_FIELD

      base.dump();

      printf("Configure done\n");

      _m.start();
      _setState(Configure);
    }

    void PVCtrls::reset() {
      QABase& base = *reinterpret_cast<QABase*>((char*)_m.reg()+0x80000);
      base.resetFbPLL();
      usleep(1000000);
      base.resetFb ();
      base.resetDma();
      usleep(1000000);
    }

    void PVCtrls::loopback(bool v) {
      std::vector<Pgp*> pgp = _m.pgp();
      for(unsigned i=0; i<4; i++)
        pgp[i]->loopback(v);

      // for(unsigned i=0; i<4; i++)
      //   pgp[i]._rxReset = 1;
      // usleep(10);
      // for(unsigned i=0; i<4; i++)
      //   pgp[i]._rxReset = 0;
      // usleep(100);
    }

    void PVCtrls::interleave(bool v) { _interleave = v; }

    void PVCtrls::_setState(Action a) {
      unsigned v = (a==Configure) ? 1 : 0;
      _state_pv->putFrom(v);
    }
  };
};
