#include "psdaq/xpm/PVPStats.hh"
#include "psdaq/xpm/Module.hh"
#include "psdaq/epicstools/PVCached.hh"
using Pds_Epics::PVCached;

#include <cpsw_error.h>  // To catch a CPSW exception and continue

#include <sstream>
#include <string>
#include <vector>

#include <stdio.h>

namespace Pds {
  namespace Xpm {

    PVPStats::PVPStats(Module& dev, unsigned partition) : 
      _dev(dev), _partition(partition), _pv(0) {}
    PVPStats::~PVPStats() {}

    void PVPStats::allocate(const std::string& title,
                            const std::string& dttitle) {
      for(unsigned i=0; i<_pv.size(); i++)
        delete _pv[i];
      _pv.resize(0);

      std::string pvbase = title + ":";
      std::string dtbase = dttitle + ":";

      _pv.push_back( new PVCached((pvbase+"L0InpRate").c_str()) );
      _pv.push_back( new PVCached((pvbase+"L0AccRate").c_str()) );
      _pv.push_back( new PVCached((pvbase+"L1Rate"   ).c_str()) );
      _pv.push_back( new PVCached((pvbase+"NumL0Inp" ).c_str()) );
      _pv.push_back( new PVCached((pvbase+"NumL0Acc" ).c_str()) );
      _pv.push_back( new PVCached((pvbase+"NumL1"    ).c_str()) );
      _pv.push_back( new PVCached((pvbase+"DeadFrac" ).c_str()) );
      _pv.push_back( new PVCached((pvbase+"DeadTime" ).c_str()) );
      _pv.push_back( new PVCached((dtbase+"DeadFLnk" ).c_str(),32) );
      _pv.push_back( new PVCached((pvbase+"RunTime"  ).c_str()) );
      _pv.push_back( new PVCached((pvbase+"MsgDelay" ).c_str()) );

      printf("Partition PVs allocated\n");
    }

#define PVPUT(i,v)    { _pv[i]->putC(double(v)); }
#define PVPUTA(p,m,v) {                                           \
      for (unsigned i = 0; i < m; ++i) _pv[p]->putC(double(v),i); \
      _pv[p]->push(); }

    void PVPStats::update(bool enabled)
    {
      const double FID_PERIOD = 14.e-6/13.;
      try {
        _dev.setPartition(_partition);
        const L0Stats& os = _last;
        L0Stats ns(_dev.l0Stats(enabled));
        if (enabled) {
          PVPUT(9, double(ns.l0Enabled)*FID_PERIOD);
          PVPUT(10, double(_dev.getL0Delay()));
          uint64_t l0Enabled = ns.l0Enabled - os.l0Enabled;
          double dt = double(l0Enabled)*FID_PERIOD;
          uint64_t numl0     = ns.numl0    - os.numl0;
          PVPUT(0, l0Enabled ? double(numl0)/dt :0);
          unsigned numl0Acc  = ns.numl0Acc - os.numl0Acc;
          PVPUT(1, l0Enabled ? double(numl0Acc)/dt:0);
          PVPUT(3, ns.numl0    - _begin.numl0);
          PVPUT(4, ns.numl0Acc - _begin.numl0Acc);
          PVPUT(6, numl0 ? double(ns.numl0Inh - os.numl0Inh) / double(numl0) : 0);
          if (l0Enabled) {
            PVPUT (7,     double(ns.l0Inhibited - os.l0Inhibited) / double(l0Enabled));
            PVPUTA(8, 32, double(ns.linkInhEv[i]  - os.linkInhEv[i])  / double(numl0));
          }
        }
        else {
          double nfid = (double(ns.time.tv_sec - os.time.tv_sec) +
                         1.e-9 * (double(ns.time.tv_nsec) - double(os.time.tv_nsec))) / 
            FID_PERIOD;
          PVPUTA(8, 32, double(ns.linkInhTm[i]  - os.linkInhTm[i])  / nfid);
        }
        _last = ns;
      } 
      catch(CPSWError& e) {
        printf("Caught exception %s\n",e.what());
      }
    }
  };
};
