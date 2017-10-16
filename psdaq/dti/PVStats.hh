#ifndef Dti_PVStats_hh
#define Dti_PVStats_hh

#include <string>
#include <vector>

namespace Pds_Epics {
  class PVWriter;
};

namespace Pds {
  namespace Dti {

    class Stats;

    class PVStats {
    public:
      PVStats();
      ~PVStats();
    public:
      void allocate(const std::string& title);
      void update(const Stats& ni, const Stats& oi, double dt);
    private:
      std::vector<Pds_Epics::PVWriter*> _pv;
    };
  };
};

#endif
