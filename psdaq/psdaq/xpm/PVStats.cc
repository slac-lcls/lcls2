#include "psdaq/xpm/PVStats.hh"
#include "psdaq/xpm/Module.hh"

#include "psdaq/epicstools/PVWriter.hh"
using Pds_Epics::PVWriter;

#include <sstream>
#include <string>
#include <vector>

#include <stdio.h>

using Pds_Epics::PVWriter;

#define PVPUSH(s) { std::ostringstream o; o << pvbase << #s << i; \
          _pv.push_back(new PVWriter(o.str().c_str())); }
#define PVPUT(v)    if ((*it)->connected()) { *reinterpret_cast<double*>((*it)->data()) = double(v); (*it)->put(); it++; }
#define PVPUTI(v)   if ((*it)->connected()) { *reinterpret_cast<int   *>((*it)->data()) = int   (v); (*it)->put(); it++; }


namespace Pds {
  namespace Xpm {

    PVStats::PVStats() : _pv(0) {}
    PVStats::~PVStats() {}

    void PVStats::_allocTiming(const std::string& title,
                               const char* sec) {
      std::ostringstream o;
      o << title << ":" << sec << ":";
      std::string pvbase = o.str();

      _pv.push_back( new PVWriter((pvbase+"RxClks").c_str()) );
      _pv.push_back( new PVWriter((pvbase+"TxClks").c_str()) );
      _pv.push_back( new PVWriter((pvbase+"RxRsts").c_str()) );
      _pv.push_back( new PVWriter((pvbase+"CrcErrs").c_str()) );
      _pv.push_back( new PVWriter((pvbase+"RxDecErrs").c_str()) );
      _pv.push_back( new PVWriter((pvbase+"RxDspErrs").c_str()) );
      _pv.push_back( new PVWriter((pvbase+"BypassRsts").c_str()) );
      _pv.push_back( new PVWriter((pvbase+"BypassDones").c_str()) );
      _pv.push_back( new PVWriter((pvbase+"RxLinkUp").c_str()) );
      _pv.push_back( new PVWriter((pvbase+"FIDs").c_str()) );
      _pv.push_back( new PVWriter((pvbase+"SOFs").c_str()) );
      _pv.push_back( new PVWriter((pvbase+"EOFs").c_str()) );
    }

    void PVStats::_updateTiming(const TimingCounts& nc,
                                const TimingCounts& oc,
                                double dt,
                                std::vector<PVWriter*>::iterator& it) {
      PVPUT(double(nc.rxClkCount       - oc.rxClkCount      ) / dt * 16e-6);
      PVPUT(double(nc.txClkCount       - oc.txClkCount      ) / dt * 16e-6);
      PVPUT(double(nc.rxRstCount       - oc.rxRstCount      ) / dt);
      PVPUT(double(nc.crcErrCount      - oc.crcErrCount     ) / dt);
      PVPUT(double(nc.rxDecErrCount    - oc.rxDecErrCount   ) / dt);
      PVPUT(double(nc.rxDspErrCount    - oc.rxDspErrCount   ) / dt);
      PVPUT(double(nc.bypassResetCount - oc.bypassResetCount) / dt);
      PVPUT(double(nc.bypassDoneCount  - oc.bypassDoneCount ) / dt);
      PVPUT(double(nc.rxLinkUp                              )     );
      PVPUT(double(nc.fidCount         - oc.fidCount        ) / dt);
      PVPUT(double(nc.sofCount         - oc.sofCount        ) / dt);
      PVPUT(double(nc.eofCount         - oc.eofCount        ) / dt);
    }

    void PVStats::allocate(const std::string& title) {
      if (ca_current_context() == NULL) {
        printf("Initializing context\n");
        SEVCHK ( ca_context_create(ca_enable_preemptive_callback ),
                 "Calling ca_context_create" );
      }

      for(unsigned i=0; i<_pv.size(); i++)
        delete _pv[i];
      _pv.resize(0);

      _allocTiming(title,"Us");
      _allocTiming(title,"Cu");

      std::ostringstream o;
      o << title << ":";
      std::string pvbase = o.str();

      for(unsigned i=0; i<32; i++) {
        PVPUSH(LinkTxReady);
        PVPUSH(LinkRxReady);
        PVPUSH(LinkTxResetDone);
        PVPUSH(LinkRxResetDone);
        PVPUSH(LinkRxRcv);
        PVPUSH(LinkRxErr);
        PVPUSH(LinkIsXpm);
	PVPUSH(RemoteLinkId);
      }
      _pv.push_back( new PVWriter((pvbase+"RecClk").c_str()) );
      _pv.push_back( new PVWriter((pvbase+"FbClk").c_str()) );
      _pv.push_back( new PVWriter((pvbase+"BpClk").c_str()) );
      printf("PVs allocated\n");
    }

    void PVStats::update(const CoreCounts& nc, const CoreCounts& oc, 
                         const LinkStatus* nl, const LinkStatus* ol,
                         unsigned recClk,
                         unsigned fbClk,
                         unsigned bpClk,
                         double dt)
    {
      std::vector<PVWriter*>::iterator it = _pv.begin();
      _updateTiming(nc.us,oc.us,dt,it);
      _updateTiming(nc.cu,oc.cu,dt,it);
      for(unsigned i=0; i<32; i++) {
        PVPUTI( nl[i].txReady );
        PVPUTI( nl[i].rxReady );
        PVPUTI( nl[i].txResetDone );
        PVPUTI( nl[i].rxResetDone );
        PVPUTI( (nl[i].rxRcvs - ol[i].rxRcvs) );
        PVPUTI( (nl[i].rxErrs - ol[i].rxErrs) );
        PVPUTI( nl[i].isXpm );
	PVPUTI( nl[i].remoteLinkId );
      }
      PVPUT(double(recClk)*1.e-6);
      PVPUT(double(fbClk )*1.e-6);
      PVPUT(double(bpClk )*1.e-6);
      //      ca_flush_io();  // Let timer do it
    }
  };
};
