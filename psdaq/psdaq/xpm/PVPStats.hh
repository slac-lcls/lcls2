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
      PVPStats(Module&, unsigned partition);
      ~PVPStats();
    public:
      void allocate(const std::string& title,
                    const std::string& dttitle);
      void update();
    private:
      Module&                           _dev;
      unsigned                          _partition;
      std::vector<Pds_Epics::PVWriter*> _pv;
      L0Stats                           _begin;
      L0Stats                           _last;
    };
  };
};

#endif
