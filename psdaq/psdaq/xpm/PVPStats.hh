#ifndef Xpm_PVPStats_hh
#define Xpm_PVPStats_hh

#include <string>
#include <vector>

#include "psdaq/xpm/Module.hh"

namespace Pds_Epics {
  class PVWriter;
};

namespace Pds {
  namespace Xpm {
    class PVPStats {
    public:
      PVPStats();
      ~PVPStats();
    public:
      void allocate(const std::string& base);
      void begin (const L0Stats& s);
      void update(const L0Stats& ns, const L0Stats& os);
    private:
      std::vector<Pds_Epics::PVWriter*> _pv;
      L0Stats                           _begin;
    };
  };
};

#endif
