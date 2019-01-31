#include "psdaq/hsd/PVCtrls.hh"
#include "psdaq/hsd/Module.hh"
#include "psdaq/hsd/FexCfg.hh"
#include "psdaq/hsd/QABase.hh"
#include "psdaq/hsd/Pgp.hh"
#include "psdaq/epicstools/EpicsCA.hh"
#include "psdaq/epicstools/PvServer.hh"
#include "psdaq/service/Task.hh"
#include "psdaq/service/Routine.hh"

#include <algorithm>
#include <sstream>
#include <cctype>
#include <stdio.h>
#include <unistd.h>

using Pds_Epics::EpicsCA;
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
        case Unconfigure: _pvc.disable  (); break;
        case EnableTr   : _pvc.enable   (); break;
        case DisableTr  : _pvc.disable  (); break;
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

#define TOU(value)  *reinterpret_cast<unsigned*>(value)
#define TON(value,e)  reinterpret_cast<unsigned*>(value)[e]

#define CPV(name, updatedBody, connectedBody)                           \
                                                                        \
    class PV(name) : public EpicsCA,                                    \
                     public PVMonitorCb                                 \
    {                                                                   \
    public:                                                             \
      PV(name)(PVCtrls& ctrl, const char* pvName) :                     \
        EpicsCA(pvName, this),                                          \
        _ctrl(ctrl) {}                                                  \
      virtual ~PV(name)() {}                                            \
    public:                                                             \
      void updated();                                                   \
      void connected(bool);                                             \
    public:                                                             \
      void put() { if (this->EpicsCA::connected())  _channel.put(); }   \
    private:                                                            \
      PVCtrls& _ctrl;                                                   \
    };                                                                  \
    void PV(name)::updated()                                            \
    {                                                                   \
      updatedBody                                                       \
    }                                                                   \
    void PV(name)::connected(bool c)                                    \
    {                                                                   \
      this->EpicsCA::connected(c);                                      \
      connectedBody                                                     \
    }
    
    CPV(ApplyConfig ,{if (TOU(data())) _ctrl.call(Configure  );}, {})
    CPV(EnableTr    ,{if (TOU(data())) _ctrl.call(EnableTr   );}, {})
    CPV(DisableTr   ,{if (TOU(data())) _ctrl.call(DisableTr  );}, {})
    CPV(UndoConfig  ,{if (TOU(data())) _ctrl.call(Unconfigure);}, {})
    CPV(Reset       ,{if (TOU(data())) _ctrl.call(Reset      );}, {})
    CPV(PgpLoopback ,{_ctrl.loopback (TOU(data())!=0);   }, {})

    PVCtrls::PVCtrls(Module& m, Pds::Task& t) : _pv(0), _m(m), _task(t) {}
    PVCtrls::~PVCtrls() {}

    void PVCtrls::call(Action a) { _task.call(new PVC_Routine(*this, a)); }

    void PVCtrls::allocate(const std::string& title)
    {
      if (ca_current_context() == NULL) {
        printf("Initializing context\n");
        SEVCHK ( ca_context_create(ca_enable_preemptive_callback ),
                 "Calling ca_context_create" );
      }

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
      NPV1(Nat_Start);
      NPV1(Nat_Gate);
      NPV1(Nat_PS);
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
      NPV(EnableTr   ,"BASE:ENABLETR");
      NPV(DisableTr  ,"BASE:DISABLETR");
      NPV(UndoConfig ,"BASE:UNDOCONFIG");
      NPV(Reset      ,"RESET");
      NPV(PgpLoopback,"PGPLOOPBACK");

      // Wait for monitors to be established
      ca_pend_io(0);
    }

    //  enumeration of PV insert order above
    enum PvIndex { Enable, 
                   Raw_Start, Raw_Gate, Raw_PS, 
                   Fex_Start, Fex_Gate, Fex_PS, 
                   Fex_Ymin, Fex_Ymax, Fex_Xpre, Fex_Xpost,
                   Nat_Start, Nat_Gate, Nat_PS,
                   FullEvt, FullSize,
                   TestPattern, SyncELo, SyncEHi, SyncOLo, SyncOHi, 
                   TrigShift, PgpSkpIntvl,
                   IntTrigVal, IntAFullVal, Partition, LastPv };

    Module& PVCtrls::module() { return _m; }

    void PVCtrls::enable() {
    }

    void PVCtrls::disable() {
    }

    void PVCtrls::configure() {
      _m.stop();

      // Update all necessary PVs
      for(unsigned i=0; i<LastPv; i++) {
        PvServer* pv = static_cast<PvServer*>(_pv[i]);
        while (!pv->EpicsCA::connected()) {
          printf("pv[%u] not connected\n",i);
          usleep(100000);
        }
        pv->update();
      }
      ca_pend_io(0);

      for(unsigned i=0; i<LastPv; i++) {
        EpicsCA* pv = _pv[i];
        printf("pv[%u] :",i);
        unsigned* q = reinterpret_cast<unsigned*>(_pv[i]->data());
        for(unsigned j=0; j<pv->data_size()/4; j++ )
          printf(" %u", q[j]);
        printf("\n");
      }

      { unsigned v = *reinterpret_cast<unsigned*>(_pv[PgpSkpIntvl]->data());
        std::vector<Pgp*> pgp = _m.pgp();
        for(unsigned i=0; i<4; i++)
          pgp[i]->skip_interval(v);
      }

      _m.dumpPgp();

      // set trigger shift
      int shift = *reinterpret_cast<int*>(_pv[TrigShift]->data());
      _m.trig_shift(shift);

      // don't know how often this is necessary
      _m.train_io(0);

      unsigned channelMask = 0;
      for(unsigned i=0; i<4; i++) {
        if (TON(_pv[Enable]->data(),i)) {
          channelMask |= (1<<i);
          if (_interleave) break;
        }
      }
      printf("channelMask 0x%x: ilv %c\n",
             channelMask, _interleave?'T':'F');
      _m.setAdcMux( _interleave, channelMask );

      // set testpattern
      int pattern = *reinterpret_cast<int*>(_pv[TestPattern]->data());
      printf("Pattern: %d\n",pattern);
      _m.disable_test_pattern();
      if (pattern>=0)
        _m.enable_test_pattern((Module::TestPattern)pattern);

      int ephlo = *reinterpret_cast<int*>(_pv[SyncELo]->data());
      int ephhi = *reinterpret_cast<int*>(_pv[SyncEHi]->data());
      int ophlo = *reinterpret_cast<int*>(_pv[SyncOLo]->data());
      int ophhi = *reinterpret_cast<int*>(_pv[SyncOHi]->data());
      while(1) {
        int eph = _m.trgPhase()[0];
        int oph = _m.trgPhase()[1];
        printf("trig phase %d [%d/%d] %d [%d/%d]\n",
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

      unsigned partition = TOU(_pv[Partition]->data());
      _m.trig_daq(partition);

      // configure fex's for each channel
      unsigned fullEvt  = TOU(_pv[FullEvt ]->data());
      unsigned fullSize = TOU(_pv[FullSize]->data());
      for(unsigned i=0; i<4; i++) {
        FexCfg& fex = _m.fex()[i];
        if ((1<<i)&channelMask) {
          unsigned streamMask=0;
          if (TON(_pv[Raw_PS]->data(),i)) {
            streamMask |= (1<<0);
            fex._base[0].setGate(4,TON(_pv[Raw_Gate]->data(),i));
            fex._base[0].setFull(fullSize,fullEvt);
            fex._base[0]._prescale=TON(_pv[Raw_PS]->data(),i)-1;
          }
          if (TON(_pv[Fex_PS]->data(),i)) {
            streamMask |= (1<<1);
            fex._base[1].setGate(4,TON(_pv[Fex_Gate]->data(),i));
            fex._base[1].setFull(0xc00,4);
            fex._base[1]._prescale=TON(_pv[Fex_PS]->data(),i)-1;
            fex._stream[1].parms[0].v=TON(_pv[Fex_Ymin]->data(),i);
            fex._stream[1].parms[1].v=TON(_pv[Fex_Ymax]->data(),i);
            fex._stream[1].parms[2].v=TON(_pv[Fex_Xpre]->data(),i);
            fex._stream[1].parms[3].v=TON(_pv[Fex_Xpost]->data(),i);
          }
          if (TON(_pv[Nat_PS]->data(),i)) {
            streamMask |= (1<<2);
            fex._base[2].setGate(4,TON(_pv[Nat_Gate]->data(),i));
            fex._base[2].setFull(0xc00,4);
            fex._base[2]._prescale=TON(_pv[Nat_PS]->data(),i)-1;
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
  };
};
