#ifndef Xpm_PVStats_hh
#define Xpm_PVStats_hh

#include <string>
#include <vector>

#include "psdaq/xpm/Module.hh"

namespace Pds_Epics {
  class PVWriter;
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
                  double dt);
    private:
      std::vector<Pds_Epics::PVWriter*> _pv;
      L0Stats _begin;
    };
  };
};

#endif
