#ifndef Hsd_PVStats_hh
#define Hsd_PVStats_hh

#include <string>
#include <vector>

namespace Pds_Epics {
  class PVWriter;
};

namespace Pds {
  namespace HSD {
    class Module;
    class Pgp;

    class PVStats {
    public:
      PVStats(Module&);
      ~PVStats();
    public:
      void allocate(const std::string& title);
      void update();
    private:
      Module& _m;
      std::vector<Pgp*> _pgp;
      std::vector<Pds_Epics::PVWriter*> _pv;
      class PVCache {
      public:
        unsigned value[4];
      };
      std::vector<PVCache> _v;
    };
  };
};

#endif
