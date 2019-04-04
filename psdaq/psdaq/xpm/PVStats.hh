#ifndef Xpm_PVStats_hh
#define Xpm_PVStats_hh

#include <string>
#include <vector>

#include "psdaq/xpm/Module.hh"

namespace Pds_Epics {
  class PVCached;
};

namespace Pds {
  class Semaphore;
  namespace Xpm {

    class PVStats {
    public:
      PVStats(Module&, Semaphore&);
      ~PVStats();
    public:
      void allocate(const std::string& title);
      void update();
      void update(const CoreCounts& nc, const CoreCounts& oc, 
                  const LinkStatus* nl, const LinkStatus* ol,
                  const PllStats* pll,
                  unsigned recClk,
                  unsigned fbClk,
                  unsigned bpClk,
                  double dt);
    private:
      void _allocTiming (const std::string&, const char*);
      void _allocPll    (const std::string&, unsigned);
      void _updateTiming(const TimingCounts& nc, 
                         const TimingCounts& oc,
                         double dt,
                         std::vector<Pds_Epics::PVCached*>::iterator&);
      void _updatePll   (const PllStats&,
                         std::vector<Pds_Epics::PVCached*>::iterator&);
    private:
      Module&    _dev;
      Semaphore& _sem;
      std::vector<Pds_Epics::PVCached*> _pv;
      timespec   _t;
      CoreCounts _c;
      LinkStatus _links[32];
    };
  };
};

#endif
