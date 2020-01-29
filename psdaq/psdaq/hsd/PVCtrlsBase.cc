#include "PVCtrlsBase.hh"
#include "FexCfg.hh"
#include "Pgp.hh"

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

namespace Pds {
  namespace HSD {

    class PVC_Routine : public Routine {
    public:
      PVC_Routine(PVCtrlsBase& pvc, Action a) : _pvc(pvc), _a(a) {}
      void routine() {
        switch(_a) {
        case ConfigureA : _pvc.configure(0); break;
        case ConfigureB : _pvc.configure(1); break;
        case Reset      : _pvc.reset     (); break;
        default: break;
        }
      }
    private:
      PVCtrlsBase& _pvc;
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
      PV(name)(PVCtrlsBase& ctrl, const char* pvName) :                  \
        EpicsPVA(pvName, this),                                         \
        _ctrl(ctrl) {}                                                  \
      virtual ~PV(name)() {}                                            \
    public:                                                             \
      virtual void updated();                                           \
      virtual void onConnect();                                         \
    public:                                                             \
      void put() { if (this->EpicsPVA::connected())  _channel.put(); }  \
    private:                                                            \
      PVCtrlsBase& _ctrl;                                                \
    };                                                                  \
    void PV(name)::updated()                                            \
    {                                                                   \
      updatedBody                                                       \
        }                                                               \
    void PV(name)::onConnect()                                          \
    {                                                                   \
      connectedBody                                                     \
        }
    
    CPV(Reset       ,{_ctrl.call(Reset     );}, {})
    CPV(ConfigA     ,{_ctrl.call(ConfigureA);}, {})
    CPV(ConfigB     ,{_ctrl.call(ConfigureB);}, {})

    PVCtrlsBase::PVCtrlsBase(Pds::Task& t) : _pv(0), _task(t) 
    {}
    PVCtrlsBase::~PVCtrlsBase() {}

    void PVCtrlsBase::call(Action a) { _task.call(new PVC_Routine(*this, a)); }

    void PVCtrlsBase::allocate(const std::string& title)
    {
      for(unsigned i=0; i<_pv.size(); i++)
        delete _pv[i];
      _pv.resize(0);
      
      std::ostringstream o;
      o << title << ":";
      std::string pvbase = o.str();

#define NPV(name,pv)  {                                                 \
        printf("Connecting to pv %s\n",(pvbase+pv).c_str());            \
        _pv.push_back( new PV(name)(*this, (pvbase+pv).c_str()) ); }

      switch(_nchips()) {
      case 1:
        NPV(ConfigA    ,"CONFIG");
        _ready[0] = new EpicsPVA((pvbase+"READY").c_str());
        break;
      case 2 :
        NPV(ConfigA    ,"A:CONFIG");
        NPV(ConfigB    ,"B:CONFIG");
        _ready[0] = new EpicsPVA((pvbase+"A:READY").c_str());
        _ready[1] = new EpicsPVA((pvbase+"B:READY").c_str());
        break;
      default:
        printf("Unknown allocation of %d chips\n",_nchips());
        break;
      }

      NPV(Reset    ,"RESET");

      _allocate();

      printf("Allocate complete\n");
    }

    void PVCtrlsBase::_configure_fex(unsigned fmc, FexCfg& fex) {

#define PVGET(name) pv.getScalarAs<unsigned>(#name)
#define PVGETF(name) pv.getScalarAs<float>(#name)

      Pds_Epics::EpicsPVA& pv = *_pv[fmc];

      unsigned streamMask=0;

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
      fex._streams= streamMask | (fullEvt<<4);
    
#define PRINT_FEX_FIELD(title,arg,op) {         \
        printf("%12.12s:",title);               \
        for(unsigned j=0; streamMask>>j; j++)   \
          printf("%c%u",                        \
                 j==0?' ':'/',                  \
                 fex._base[j].arg op);          \
      }                                         \
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
    }

  };
};
