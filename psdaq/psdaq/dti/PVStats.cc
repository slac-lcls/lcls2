#include "psdaq/dti/PVStats.hh"
#include "psdaq/dti/Module.hh"

#include "psdaq/epicstools/EpicsPVA.hh"

#include <sstream>
#include <string>
#include <vector>

#include <stdio.h>

using Pds_Epics::EpicsPVA;

enum { _TimLinkUp,
       _TimRefClk,
       _TimFrRate,
       _UsLinkUp,
       _BpLinkUp,
       _DsLinkUp,
       _UsRxErrs,
       _dUsRxErrs,
       _UsRxFull,
       _dUsRxFull,
       _UsRxInh,
       _dUsRxInh,
       _UsWrFifoD,
       _dUsWrFifoD,
       _UsRdFifoD,
       _dUsRdFifoD,
       _UsIbEvt,
       _dUsIbEvt,
       _UsObRecv,
       _dUsObRecv,
       _UsObSent,
       _dUsObSent,
       _BpObSent,
       _dBpObSent,
       _DsRxErrs,
       _dDsRxErrs,
       _DsRxFull,
       _dDsRxFull,
       _DsObSent,
       _dDsObSent,
       _QpllLock,
       _MonClkRate,
       //       _MonClkSlow,
       //       _MonClkFast,
       //       _MonClkLock,
       _UsLinkObL0,
       _dUsLinkObL0,
       _UsLinkObL1A,
       _dUsLinkObL1A,
       _UsLinkObL1R,
       _dUsLinkObL1R,

       _UsLinkMsgDelay,
       _PartMsgDelay,
       _NumberOf,
};

namespace Pds {
  namespace Dti {

    PVStats::PVStats() : _pv(_NumberOf) {}
    PVStats::~PVStats() {}

    void PVStats::allocate(const std::string& title) {
      for(unsigned i=0; i<_NumberOf; i++)
        if (_pv[i]) {
          delete _pv[i];
          _pv[i]=0;
        }

      std::ostringstream o;
      o << title << ":";
      std::string pvbase = o.str();

#define PV_ADD(name  ) { _pv[_##name] = new EpicsPVA((pvbase + #name).c_str()); }
#define PV_ADDV(name,n) { _pv[_##name] = new EpicsPVA((pvbase + #name).c_str(), n); }

      PV_ADD (TimLinkUp);
      PV_ADD (TimRefClk);
      PV_ADD (TimFrRate);
      PV_ADD (UsLinkUp);
      PV_ADD (BpLinkUp);
      PV_ADD (DsLinkUp);
      PV_ADDV(UsRxErrs  ,Module::NUsLinks);
      PV_ADDV(dUsRxErrs ,Module::NUsLinks);
      PV_ADDV(UsRxFull  ,Module::NUsLinks);
      PV_ADDV(dUsRxFull ,Module::NUsLinks);
      // PV_ADDV(UsIbRecv  ,Module::NUsLinks);
      // PV_ADDV(dUsIbRecv ,Module::NUsLinks);
      PV_ADDV(UsRxInh   ,Module::NUsLinks);
      PV_ADDV(dUsRxInh  ,Module::NUsLinks);
      PV_ADDV(UsWrFifoD ,Module::NUsLinks);
      PV_ADDV(dUsWrFifoD,Module::NUsLinks);
      PV_ADDV(UsRdFifoD ,Module::NUsLinks);
      PV_ADDV(dUsRdFifoD,Module::NUsLinks);
      PV_ADDV(UsIbEvt   ,Module::NUsLinks);
      PV_ADDV(dUsIbEvt  ,Module::NUsLinks);
      PV_ADDV(UsObRecv  ,Module::NUsLinks);
      PV_ADDV(dUsObRecv ,Module::NUsLinks);
      PV_ADDV(UsObSent  ,Module::NUsLinks);
      PV_ADDV(dUsObSent ,Module::NUsLinks);

      PV_ADD (BpObSent  );
      PV_ADD (dBpObSent );

      PV_ADDV(DsRxErrs  ,Module::NDsLinks);
      PV_ADDV(dDsRxErrs ,Module::NDsLinks);
      PV_ADDV(DsRxFull  ,Module::NDsLinks);
      PV_ADDV(dDsRxFull ,Module::NDsLinks);
      PV_ADDV(DsObSent  ,Module::NDsLinks);
      PV_ADDV(dDsObSent ,Module::NDsLinks);

      PV_ADD (QpllLock);
      PV_ADDV(MonClkRate,4);

      PV_ADDV(UsLinkObL0  ,Module::NUsLinks);
      PV_ADDV(dUsLinkObL0 ,Module::NUsLinks);
      PV_ADDV(UsLinkObL1A ,Module::NUsLinks);
      PV_ADDV(dUsLinkObL1A,Module::NUsLinks);
      PV_ADDV(UsLinkObL1R ,Module::NUsLinks);
      PV_ADDV(dUsLinkObL1R,Module::NUsLinks);

      PV_ADDV(UsLinkMsgDelay,Module::NUsLinks);
      PV_ADDV(PartMsgDelay  ,8);
#undef PV_ADD
#undef PV_ADDV

      printf("PVs allocated\n");
    }

    void PVStats::update(const Stats& ns, const Stats& os, double dt)
    {
#define PVPUTU(i,v)    { _pv[i]->putFrom<unsigned>(unsigned(v+0.5)); }
#define PVPUTD(i,v)    { _pv[i]->putFrom<double>(double(v)); }
#define PVPUTAU(p,m,v) { pvd::shared_vector<unsigned> vec(m);                                    \
                         for (unsigned i = 0; i < m; ++i) vec[i] = unsigned(v+0.5);           \
                         _pv[p]->putFromVector<unsigned>(freeze(vec));                                      \
                       }
#define PVPUTAD(p,m,v) { pvd::shared_vector<double> vec(m);                                    \
                         for (unsigned i = 0; i < m; ++i) vec[i] = double  (v);             \
                         _pv[p]->putFromVector<double>(freeze(vec));                                      \
                       }

      PVPUTU ( _TimLinkUp, ns.timLinkUp);
      PVPUTD ( _TimRefClk, ((ns.timRefCount-os.timRefCount)*16/dt));
      PVPUTD ( _TimFrRate, (ns.timFrCount -os.timFrCount) / dt);

      PVPUTU ( _UsLinkUp, ns.usLinkUp);
      PVPUTU ( _BpLinkUp, ns.bpLinkUp);
      PVPUTU ( _DsLinkUp, ns.dsLinkUp);

#define PVPUT_ABS( idx, elm ) {                                                     \
        pvd::shared_vector<unsigned> vec(Module::NUsLinks);                         \
        for(unsigned i=0; i<Module::NUsLinks; i++) vec[i] = ns.us[i].elm;           \
        _pv[idx]->putFromVector<unsigned>(freeze(vec));   }

#define PVPUT_DEL( idx, elm ) {                                                        \
        pvd::shared_vector<unsigned> vec(Module::NUsLinks);                            \
        for(unsigned i=0; i<Module::NUsLinks; i++) vec[i] = ns.us[i].elm-os.us[i].elm; \
        _pv[idx]->putFromVector<unsigned>(freeze(vec));   }


      PVPUT_ABS( _UsRxErrs , rxErrs);
      PVPUT_DEL( _dUsRxErrs, rxErrs);
      PVPUT_ABS( _UsRxFull , rxFull);
      //      PVPUT_DEL( _dUsRxFull, rxFull);
      {
          pvd::shared_vector<double> vec(Module::NUsLinks);
          for(unsigned i=0; i<Module::NUsLinks; i++) vec[i] = (ns.us[i].rxFull-os.us[i].rxFull)/156.25e6;
         _pv[_dDsRxFull]->putFromVector<double>(freeze(vec));
      }
      //      PVPUTAU( 7, Module::NUsLinks,        ns.us[i].ibRecv);
      //      PVPUTAD( 8, Module::NUsLinks, double(ns.us[i].ibRecv - os.us[i].ibRecv) / dt);
      PVPUT_ABS( _UsRxInh  , rxInh);
      PVPUT_DEL( _dUsRxInh , rxInh);
      PVPUT_ABS( _UsWrFifoD  , wrFifoD);
      PVPUT_DEL( _dUsWrFifoD , wrFifoD);
      PVPUT_ABS( _UsRdFifoD  , rdFifoD);
      PVPUT_DEL( _dUsRdFifoD , rdFifoD);

      PVPUT_ABS( _UsIbEvt  , ibEvt);
      PVPUT_DEL( _dUsIbEvt , ibEvt);
      PVPUT_ABS( _UsObRecv , obRecv);
      PVPUT_DEL( _dUsObRecv, obRecv);
      PVPUT_ABS( _UsObSent , obSent);
      PVPUT_DEL( _dUsObSent, obSent);

      /*
      PVPUTU ( _UsLinkObL0  ,        ns.usLinkObL0);
      PVPUTU ( _dUsLinkObL0 , double(ns.usLinkObL0  - os.usLinkObL0)  / dt);
      PVPUTU ( _UsLinkObL1A ,        ns.usLinkObL1A);
      PVPUTU ( _dUsLinkObL1A, double(ns.usLinkObL1A - os.usLinkObL1A) / dt);
      PVPUTU ( _UsLinkObL1R ,        ns.usLinkObL1R);
      PVPUTU ( _dUsLinkObL1R, double(ns.usLinkObL1R - os.usLinkObL1R) / dt);
      */

      PVPUTU ( _BpObSent ,        ns.bpObSent);
      PVPUTU ( _dBpObSent, double(ns.bpObSent - os.bpObSent) / dt);

#undef PVPUT_ABS
#undef PVPUT_DEL

#define PVPUT_ABS( idx, elm ) {                                                    \
        pvd::shared_vector<unsigned> vec(Module::NDsLinks);                                          \
        for(unsigned i=0; i<Module::NDsLinks; i++) vec[i] = ns.ds[i].elm;         \
        _pv[idx]->putFromVector<unsigned>(freeze(vec));                                            \
    }

#define PVPUT_DEL( idx, elm ) {                                         \
        pvd::shared_vector<unsigned> vec(Module::NDsLinks);                                          \
        for(unsigned i=0; i<Module::NDsLinks; i++) vec[i] = ns.ds[i].elm-os.ds[i].elm; \
        _pv[idx]->putFromVector<unsigned>(freeze(vec));                                            \
    }

      PVPUT_ABS( _DsRxErrs , rxErrs);
      PVPUT_DEL( _dDsRxErrs, rxErrs);
      PVPUT_ABS( _DsRxFull , rxFull);
      //      PVPUT_DEL( _dDsRxFull, rxFull);
      {
          pvd::shared_vector<double> vec(Module::NDsLinks);
          for(unsigned i=0; i<Module::NDsLinks; i++)
            vec[i] = (ns.ds[i].rxFull-os.ds[i].rxFull)/156.25e6;
          _pv[_dDsRxFull]->putFromVector<double>(freeze(vec));
      }
      PVPUT_ABS( _DsObSent , obSent);
      //      PVPUT_DEL( _dDsObSent, obSent);
      {
          pvd::shared_vector<double> vec(Module::NDsLinks);
          for(unsigned i=0; i<Module::NDsLinks; i++)
            vec[i] = double(ns.ds[i].obSent-os.ds[i].obSent)*8.e-6;
          _pv[_dDsObSent]->putFromVector<double>(freeze(vec));
      }

      PVPUTU ( _QpllLock , ns.qpllLock);

      PVPUTAU( _MonClkRate, 4,  ns.monClk[i].rate);

      {
          pvd::shared_vector<unsigned> vec(Module::NUsLinks);
          for(unsigned i=0; i<Module::NUsLinks; i++) vec[i] = ns.usLinkMsgDelay[i];
          _pv[_UsLinkMsgDelay]->putFromVector<unsigned>(freeze(vec));
      }

      {
          pvd::shared_vector<unsigned> vec(8);
          for(unsigned i=0; i<8; i++) vec[i] = ns.partMsgDelay[i];
          _pv[_PartMsgDelay]->putFromVector<unsigned>(freeze(vec));
      }

#undef PVPUT_ABS
#undef PVPUT_DEL
#undef PVPUTU
#undef PVPUTD
#undef PVPUTAU
#undef PVPUTAD

    }
  };
};
