#ifndef Xpm_PVStats_hh
#define Xpm_PVStats_hh

#include <string>
#include <vector>

#include "psdaq/xpm/Module.hh"

namespace Pds_Epics {
  class EpicsPVA;
};

namespace Pds {
  namespace Xpm {

    class PVStats {
    public:
      PVStats();
      ~PVStats();
    public:
      void allocate(const std::string& title);
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
                         std::vector<Pds_Epics::EpicsPVA*>::iterator&);
      void _updatePll   (const PllStats&,
                         std::vector<Pds_Epics::EpicsPVA*>::iterator&);
    private:
      std::vector<Pds_Epics::EpicsPVA*> _pv;
      L0Stats _begin;
    };
  };
};

#endif
