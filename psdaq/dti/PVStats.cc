#include "psdaq/dti/PVStats.hh"
#include "psdaq/dti/Module.hh"

#include "psdaq/epicstools/PVWriter.hh"
using Pds_Epics::PVWriter;

#include <sstream>
#include <string>
#include <vector>

#include <stdio.h>

using Pds_Epics::PVWriter;

namespace Pds {
  namespace Dti {

    PVStats::PVStats() : _pv(0) {}
    PVStats::~PVStats() {}

    void PVStats::allocate(const std::string& title) {
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

      _pv.push_back( new PVWriter((pvbase+"UsLinkUp").c_str()) );
      _pv.push_back( new PVWriter((pvbase+"BpLinkUp" ).c_str()) );
      _pv.push_back( new PVWriter((pvbase+"DsLinkUp").c_str()) );

      _pv.push_back( new PVWriter((pvbase+ "UsRxErrs").c_str(), Module::NUsLinks) );
      _pv.push_back( new PVWriter((pvbase+"dUsRxErrs").c_str(), Module::NUsLinks) );
      _pv.push_back( new PVWriter((pvbase+ "UsRxFull").c_str(), Module::NUsLinks) );
      _pv.push_back( new PVWriter((pvbase+"dUsRxFull").c_str(), Module::NUsLinks) );
      _pv.push_back( new PVWriter((pvbase+ "UsIbRecv").c_str(), Module::NUsLinks) );
      _pv.push_back( new PVWriter((pvbase+"dUsIbRecv").c_str(), Module::NUsLinks) );
      _pv.push_back( new PVWriter((pvbase+ "UsIbEvt" ).c_str(), Module::NUsLinks) );
      _pv.push_back( new PVWriter((pvbase+"dUsIbEvt" ).c_str(), Module::NUsLinks) );
      _pv.push_back( new PVWriter((pvbase+ "UsObRecv").c_str(), Module::NUsLinks) );
      _pv.push_back( new PVWriter((pvbase+"dUsObRecv").c_str(), Module::NUsLinks) );
      _pv.push_back( new PVWriter((pvbase+ "UsObSent").c_str(), Module::NUsLinks) );
      _pv.push_back( new PVWriter((pvbase+"dUsObSent").c_str(), Module::NUsLinks) );

      _pv.push_back( new PVWriter((pvbase+ "BpObSent").c_str()) );
      _pv.push_back( new PVWriter((pvbase+"dBpObSent").c_str()) );

      _pv.push_back( new PVWriter((pvbase+ "DsRxErrs").c_str(), Module::NDsLinks) );
      _pv.push_back( new PVWriter((pvbase+"dDsRxErrs").c_str(), Module::NDsLinks) );
      _pv.push_back( new PVWriter((pvbase+ "DsRxFull").c_str(), Module::NDsLinks) );
      _pv.push_back( new PVWriter((pvbase+"dDsRxFull").c_str(), Module::NDsLinks) );
      _pv.push_back( new PVWriter((pvbase+ "DsObSent").c_str(), Module::NDsLinks) );
      _pv.push_back( new PVWriter((pvbase+"dDsObSent").c_str(), Module::NDsLinks) );

      _pv.push_back( new PVWriter((pvbase+"QpllLock").c_str()) );

      _pv.push_back( new PVWriter((pvbase+"MonClkRate").c_str(), 4) );
      _pv.push_back( new PVWriter((pvbase+"MonClkSlow").c_str(), 4) );
      _pv.push_back( new PVWriter((pvbase+"MonClkFast").c_str(), 4) );
      _pv.push_back( new PVWriter((pvbase+"MonClkLock").c_str(), 4) );

      _pv.push_back( new PVWriter((pvbase+ "UsLinkObL0" ).c_str()) );
      _pv.push_back( new PVWriter((pvbase+"dUsLinkObL0" ).c_str()) );
      _pv.push_back( new PVWriter((pvbase+ "UsLinkObL1A").c_str()) );
      _pv.push_back( new PVWriter((pvbase+"dUsLinkObL1A").c_str()) );
      _pv.push_back( new PVWriter((pvbase+ "UsLinkObL1R").c_str()) );
      _pv.push_back( new PVWriter((pvbase+"dUsLinkObL1R").c_str()) );

      _pv.push_back( new PVWriter((pvbase+ "RxFrErrs").c_str(), 2) );
      _pv.push_back( new PVWriter((pvbase+"dRxFrErrs").c_str(), 2) );
      _pv.push_back( new PVWriter((pvbase+ "RxFrames").c_str(), 2) );
      _pv.push_back( new PVWriter((pvbase+"dRxFrames").c_str(), 2) );
      _pv.push_back( new PVWriter((pvbase+ "RxOpCodes").c_str(), 2) );
      _pv.push_back( new PVWriter((pvbase+"dRxOpCodes").c_str(), 2) );
      _pv.push_back( new PVWriter((pvbase+ "TxFrErrs").c_str(), 2) );
      _pv.push_back( new PVWriter((pvbase+"dTxFrErrs").c_str(), 2) );
      _pv.push_back( new PVWriter((pvbase+ "TxFrames").c_str(), 2) );
      _pv.push_back( new PVWriter((pvbase+"dTxFrames").c_str(), 2) );
      _pv.push_back( new PVWriter((pvbase+ "TxOpCodes").c_str(), 2) );
      _pv.push_back( new PVWriter((pvbase+"dTxOpCodes").c_str(), 2) );

      _pv.push_back( new PVWriter((pvbase+"MyTestPV").c_str()) );

      printf("PVs allocated\n");
    }

    static unsigned myTestCounter = 0;

    void PVStats::update(const Stats& ns, const Stats& os, double dt)
    {
#define PVPUTU(i,v)    { *reinterpret_cast<unsigned*>(_pv[i]->data())    = unsigned(v); _pv[i]->put(); }
#define PVPUTD(i,v)    { *reinterpret_cast<double  *>(_pv[i]->data())    = double  (v); _pv[i]->put(); }
#define PVPUTAU(p,m,v) { for (unsigned i = 0; i < m; ++i)                                \
                           reinterpret_cast<unsigned*>(_pv[p]->data())[i] = unsigned(v); \
                         _pv[p]->put();                                                  \
                       }
#define PVPUTAD(p,m,v) { for (unsigned i = 0; i < m; ++i)                                \
                           reinterpret_cast<double  *>(_pv[p]->data())[i] = double  (v); \
                         _pv[p]->put();                                                  \
                       }

      PVPUTU ( 0, ns.usLinkUp);
      PVPUTU ( 1, ns.bpLinkUp);
      PVPUTU ( 2, ns.dsLinkUp);

      PVPUTAU( 3, Module::NUsLinks,        ns.us[i].rxErrs);
      PVPUTAD( 4, Module::NUsLinks, double(ns.us[i].rxErrs - os.us[i].rxErrs) / dt);
      PVPUTAU( 5, Module::NUsLinks,        ns.us[i].rxFull);
      PVPUTAD( 6, Module::NUsLinks, double(ns.us[i].rxFull - os.us[i].rxFull) / dt);
      PVPUTAU( 7, Module::NUsLinks,        ns.us[i].ibRecv);
      PVPUTAD( 8, Module::NUsLinks, double(ns.us[i].ibRecv - os.us[i].ibRecv) / dt);
      PVPUTAU( 9, Module::NUsLinks,        ns.us[i].ibEvt);
      PVPUTAD(10, Module::NUsLinks, double(ns.us[i].ibEvt  - os.us[i].ibEvt ) / dt);
      PVPUTAU(11, Module::NUsLinks,        ns.us[i].obRecv);
      PVPUTAD(12, Module::NUsLinks, double(ns.us[i].obRecv - os.us[i].obRecv) / dt);
      PVPUTAU(13, Module::NUsLinks,        ns.us[i].obSent);
      PVPUTAD(14, Module::NUsLinks, double(ns.us[i].obSent - os.us[i].obSent) / dt);

      PVPUTU (15,        ns.bpObSent);
      PVPUTD (16, double(ns.bpObSent - os.bpObSent) / dt);

      PVPUTAU(17, Module::NDsLinks,        ns.ds[i].rxErrs);
      PVPUTAD(18, Module::NDsLinks, double(ns.ds[i].rxErrs - os.ds[i].rxErrs) / dt);
      PVPUTAU(19, Module::NDsLinks,        ns.ds[i].rxFull);
      PVPUTAD(20, Module::NDsLinks, double(ns.ds[i].rxFull - os.ds[i].rxFull) / dt);
      PVPUTAU(21, Module::NDsLinks,        ns.ds[i].obSent);
      PVPUTAD(22, Module::NDsLinks, double(ns.ds[i].obSent - os.ds[i].obSent) / dt);

      PVPUTU (23, ns.qpllLock);

      PVPUTAU(24, 4,  ns.monClk[i].rate);
      PVPUTAU(25, 4,  ns.monClk[i].slow);
      PVPUTAU(26, 4,  ns.monClk[i].fast);
      PVPUTAU(27, 4,  ns.monClk[i].lock);

      PVPUTU (28,        ns.usLinkObL0);
      PVPUTD (29, double(ns.usLinkObL0  - os.usLinkObL0)  / dt);
      PVPUTU (30,        ns.usLinkObL1A);
      PVPUTD (31, double(ns.usLinkObL1A - os.usLinkObL1A) / dt);
      PVPUTU (32,        ns.usLinkObL1R);
      PVPUTD (33, double(ns.usLinkObL1R - os.usLinkObL1R) / dt);

      PVPUTAU(34, 2,        ns.pgp[i].rxFrameErrs);
      PVPUTAD(35, 2, double(ns.pgp[i].rxFrameErrs - os.pgp[i].rxFrameErrs) / dt);
      PVPUTAU(36, 2,        ns.pgp[i].rxFrames);
      PVPUTAD(37, 2, double(ns.pgp[i].rxFrames    - os.pgp[i].rxFrames   ) / dt);
      PVPUTAU(38, 2,        ns.pgp[i].rxOpcodes);
      PVPUTAD(39, 2, double(ns.pgp[i].rxOpcodes   - os.pgp[i].rxOpcodes  ) / dt);
      PVPUTAU(40, 2,        ns.pgp[i].txFrameErrs);
      PVPUTAD(41, 2, double(ns.pgp[i].txFrameErrs - os.pgp[i].txFrameErrs) / dt);
      PVPUTAU(42, 2,        ns.pgp[i].txFrames);
      PVPUTAD(43, 2, double(ns.pgp[i].txFrames    - os.pgp[i].txFrames   ) / dt);
      PVPUTAU(44, 2,        ns.pgp[i].txOpcodes);
      PVPUTAD(45, 2, double(ns.pgp[i].txOpcodes   - os.pgp[i].txOpcodes  ) / dt);

      PVPUTU(46, myTestCounter++);

#undef PVPUTU
#undef PVPUTD
#undef PVPUTAU
#undef PVPUTAD

      ca_flush_io();
    }
  };
};
