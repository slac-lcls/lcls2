#include "psdaq/xpm/PVStats.hh"
#include "psdaq/xpm/Module.hh"

#include "psdaq/epicstools/PVCached.hh"
using Pds_Epics::PVCached;

#include "psdaq/service/Semaphore.hh"

#include <cpsw_error.h>

#include <sstream>
#include <string>
#include <vector>

#include <stdio.h>

namespace Pds {
  namespace Xpm {

    PVStats::PVStats(Module& dev, Semaphore& sem) : _dev(dev), _sem(sem), _pv(0) {}
    PVStats::~PVStats() {}

    //#define PVPUT(v)    if ((*it)->connected()) { (*it)->putFrom<double>(double(v)); } it++;
    //#define PVPUTI(v)   if ((*it)->connected()) { (*it)->putFrom<int>(int(v)); } it++;
#define PVPUT(v)    if ((*it)->connected()) { (*it)->putC(double(v)); } it++;
#define PVPUTI(v)   PVPUT(v)

    void PVStats::_allocTiming(const std::string& title,
                               const char* sec) {
#define PVPUSH(s) _pv.push_back(new PVCached((pvbase+#s).c_str()))
      std::ostringstream o;
      o << title << ":" << sec << ":";
      std::string pvbase = o.str();
      PVPUSH(RxClks);
      PVPUSH(TxClks);
      PVPUSH(RxRsts);
      PVPUSH(CrcErrs);
      PVPUSH(RxDecErrs);
      PVPUSH(RxDspErrs);
      PVPUSH(BypassRsts);
      PVPUSH(BypassDones);
      PVPUSH(RxLinkUp);
      PVPUSH(FIDs);
      PVPUSH(SOFs);
      PVPUSH(EOFs);
#undef PVPUSH
    }

    void PVStats::_updateTiming(const TimingCounts& nc,
                                const TimingCounts& oc,
                                double dt,
                                std::vector<PVCached*>::iterator& it) {
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

    void PVStats::_allocPll(const std::string& title,
                            unsigned amc) {
#define PVPUSH(s) { std::ostringstream o; o << title << ":" << #s << amc;      \
        _pv.push_back(new PVCached(o.str().c_str())); }
      PVPUSH(PLL_LOL);
      PVPUSH(PLL_LOLCNT);
      PVPUSH(PLL_LOS);
      PVPUSH(PLL_LOSCNT);
#undef PVPUSH
    }

    void PVStats::_updatePll(const PllStats& s,
                             std::vector<PVCached*>::iterator& it) {
      PVPUTI(s.lol);
      PVPUTI(s.lolCount);
      PVPUTI(s.los);
      PVPUTI(s.losCount);
    }

    void PVStats::allocate(const std::string& title) {
      for(unsigned i=0; i<_pv.size(); i++)
        delete _pv[i];
      _pv.resize(0);

      _allocTiming(title,"Us");
      _allocTiming(title,"Cu");
      for(unsigned i=0; i<Module::NAmcs; i++)
        _allocPll   (title,i);

      std::ostringstream o;
      o << title << ":";
      std::string pvbase = o.str();

#define PVPUSH(s) _pv.push_back(new PVCached((pvbase+"XTPG:"+#s).c_str()))
      PVPUSH(TimeStamp);
      PVPUSH(PulseId);
      PVPUSH(FiducialIntv);
      PVPUSH(FiducialErr);
#undef PVPUSH

#define PVPUSH(s) { std::ostringstream o; o << pvbase << #s << i; \
          _pv.push_back(new PVCached(o.str().c_str())); }
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
#undef PVPUSH
      _pv.push_back( new PVCached((pvbase+"RecClk").c_str()) );
      _pv.push_back( new PVCached((pvbase+"FbClk").c_str()) );
      _pv.push_back( new PVCached((pvbase+"BpClk").c_str()) );

      printf("PVs allocated %zu\n", _pv.size());
    }

    void PVStats::update()
    {
      timespec t; clock_gettime(CLOCK_REALTIME,&t);
      CoreCounts c = _dev.counts();
      LinkStatus links[32];
      _dev.linkStatus(links);
      PllStats   pll[2];
      for(unsigned i=0; i<Module::NAmcs; i++)
        pll[i] = _dev.pllStat(i);
      unsigned bpClk  = _dev._monClk[0]&0x1fffffff;
      unsigned fbClk  = _dev._monClk[1]&0x1fffffff;
      unsigned recClk = _dev._monClk[2]&0x1fffffff;
      double dt = double(t.tv_sec-_t.tv_sec)+1.e-9*(double(t.tv_nsec)-double(_t.tv_nsec));
      _sem.take();
      try {
        update(c,_c,links,_links,pll,recClk,fbClk,bpClk,dt);
      } catch (CPSWError& e) {
        printf("Caught exception %s\n", e.what());
      }
      _sem.give();
      _c=c;
      std::copy(links,links+32,_links);
      _t=t;
    }

    void PVStats::update(const CoreCounts& nc, const CoreCounts& oc,
                         const LinkStatus* nl, const LinkStatus* ol,
                         const PllStats* pll,
                         unsigned recClk,
                         unsigned fbClk,
                         unsigned bpClk,
                         double dt)
    {
      std::vector<PVCached*>::iterator it = _pv.begin();
      _updateTiming(nc.us,oc.us,dt,it);
      _updateTiming(nc.cu,oc.cu,dt,it);
      for(unsigned i=0; i<Module::NAmcs; i++)
        _updatePll(pll[i],it);

      PVPUT(uint64_t(_dev._timestamp));
      PVPUT((uint64_t(_dev._pulseId)&0x00ffffffffffffffULL));
      unsigned v = _dev._cuFiducialIntv;
      PVPUT((v & ~(1<<31)));
      PVPUT((v >> 31));

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
