#include "psdaq/xpm/PVStats.hh"
#include "psdaq/xpm/Module.hh"

#include "psdaq/epicstools/PVWriter.hh"
using Pds_Epics::PVWriter;

#include <sstream>
#include <string>
#include <vector>

#include <stdio.h>

using Pds_Epics::PVWriter;

namespace Pds {
  namespace Xpm {

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
#define PVPUSH(s) { std::ostringstream o; o << pvbase << #s << i; \
          _pv.push_back(new PVWriter(o.str().c_str())); }
      for(unsigned i=0; i<32; i++) {
        PVPUSH(LinkTxReady);
        PVPUSH(LinkRxReady);
        PVPUSH(LinkRxErr);
        PVPUSH(LinkIsXpm);
      }
#undef PVPUSH
      printf("PVs allocated\n");
    }

#define PVPUT(v)    if ((*it)->connected()) { *reinterpret_cast<double*>((*it)->data()) = double(v); (*it)->put(); it++; }
#define PVPUTI(v)   if ((*it)->connected()) { *reinterpret_cast<int   *>((*it)->data()) = int   (v); (*it)->put(); it++; }

    void PVStats::update(const CoreCounts& nc, const CoreCounts& oc, 
                         const LinkStatus* nl, const LinkStatus* ol,
                         double dt)
    {
      std::vector<PVWriter*>::iterator it = _pv.begin();
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
      for(unsigned i=0; i<32; i++) {
        PVPUTI( nl[i].txReady );
        PVPUTI( nl[i].rxReady );
        PVPUTI( (nl[i].rxErrs - ol[i].rxErrs) );
        PVPUTI( nl[i].isXpm );
      }
      //      ca_flush_io();  // Let timer do it
    }
  };
};
