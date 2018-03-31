#include "psdaq/hsd/PVCtrls.hh"
#include "psdaq/hsd/Module.hh"
#include "psdaq/hsd/FexCfg.hh"
#include "psdaq/hsd/QABase.hh"
#include "psdaq/hsd/Pgp2bAxi.hh"
#include "psdaq/epicstools/EpicsCA.hh"

#include <algorithm>
#include <sstream>
#include <cctype>
#include <stdio.h>

using Pds_Epics::EpicsCA;
using Pds_Epics::PVMonitorCb;

static std::string STOU(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c){ return std::toupper(c); }
                 );
  return s;
}

namespace Pds {
  namespace HSD {

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

    CPV(ApplyConfig ,{if (TOU(data())) _ctrl.configure();}, {})
    CPV(Reset       ,{if (TOU(data())) _ctrl.reset    ();}, {})
    CPV(PgpLoopback ,{_ctrl.loopback (TOU(data())!=0);   }, {})

    class GPV : public EpicsCA {
    public:
      GPV(const char* pvName) :
        EpicsCA(pvName, NULL) {}
    public:
      virtual ~GPV() {}
    public:
      void get() { if (connected()) _channel.get(); }
    };

    PVCtrls::PVCtrls(Module& m) : _pv(0), _m(m) {}
    PVCtrls::~PVCtrls() {}

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
#define NPV1(name)  _pv.push_back( new GPV(STOU(pvbase+#name).c_str()) )

      NPV1(Enable);
      NPV1(Raw_Gate);
      NPV1(Raw_PS);
      NPV1(Fex_Gate);
      NPV1(Fex_PS);
      NPV1(Fex_Ymin);
      NPV1(Fex_Ymax);
      NPV1(Fex_Xpre);
      NPV1(Fex_Xpost);
      NPV1(TestPattern);
      NPV1(IntTrigVal);
      NPV1(IntAFullVal);
      _pv.push_back(new GPV((pvbase+"BASE:PARTITION").c_str()));
      
      NPV(ApplyConfig,"BASE:APPLYCONFIG");
      NPV(Reset      ,"RESET");
      NPV(PgpLoopback,"PGPLOOPBACK");

      // Wait for monitors to be established
      ca_pend_io(0);
    }

    //  enumeration of PV insert order above
    enum PvIndex { Enable, Raw_Gate, Raw_PS, 
                   Fex_Gate, Fex_PS, 
                   Fex_Ymin, Fex_Ymax, Fex_Xpre, Fex_Xpost,
                   TestPattern, IntTrigVal, IntAFullVal, Partition, LastPv };

    Module& PVCtrls::module() { return _m; }

    void PVCtrls::configure() {
      _m.stop();

      // Update all necessary PVs
      for(unsigned i=0; i<LastPv; i++)
        static_cast<GPV*>(_pv[i])->get();
      ca_pend_io(0);

      // set testpattern
      int pattern = *reinterpret_cast<int*>(_pv[TestPattern]->data());
      if (pattern<0)
        _m.disable_test_pattern();
      else
        _m.enable_test_pattern((Module::TestPattern)pattern);

      // configure fex's for each channel
      unsigned channelMask = 0;
      for(unsigned i=0; i<4; i++) {
        FexCfg& fex = _m.fex()[i];
        if (TON(_pv[Enable]->data(),i)) {
          channelMask |= (1<<i);
          fex._base[0].setGate(4,TON(_pv[Raw_Gate]->data(),i));
          fex._base[0].setFull(0xc00,4);
          fex._base[0]._prescale=TON(_pv[Raw_PS]->data(),i);
          fex._base[0].setGate(4,TON(_pv[Fex_Gate]->data(),i));
          fex._base[0].setFull(0xc00,4);
          fex._base[0]._prescale=TON(_pv[Fex_PS]->data(),i);
          fex._stream[1].parms[0].v=TON(_pv[Fex_Ymin]->data(),i);
          fex._stream[1].parms[1].v=TON(_pv[Fex_Ymax]->data(),i);
          fex._stream[1].parms[2].v=TON(_pv[Fex_Xpre]->data(),i);
          fex._stream[1].parms[3].v=TON(_pv[Fex_Xpost]->data(),i);
          fex._streams= (TON(_pv[Raw_PS]->data(),i)?1:0) | 
            (TON(_pv[Fex_PS]->data(),i)?2:0);
        }
        else
          fex._streams= 0;
      }

      // set base gate and mux
      QABase& base = *reinterpret_cast<QABase*>((char*)_m.reg()+0x80000);
      base.init();
      //      _m.sample_init(32+48*length, 0, 0);
      _m.setAdcMux(false, channelMask);

      // set trig (partition)
      unsigned partition = TOU(_pv[Partition]->data());
      _m.trig_daq(partition);

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
      Pgp2bAxi* pgp = reinterpret_cast<Pgp2bAxi*>((char*)_m.reg()+0x90000);
      for(unsigned i=0; i<4; i++)
        if (v)
          pgp[i]._loopback |= 2;
        else
          pgp[i]._loopback &= ~2;

      for(unsigned i=0; i<4; i++)
        pgp[i]._rxReset = 1;
      usleep(10);
      for(unsigned i=0; i<4; i++)
        pgp[i]._rxReset = 0;
      usleep(100);
    }
  };
};
