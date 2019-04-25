#ifndef Hsd_PV64Stats_hh
#define Hsd_PV64Stats_hh

#include <string>
#include <vector>

namespace Pds_Epics {
  class EpicsPVA;
};

namespace Pds {
  namespace HSD {
    class Pgp;
    class Module64;

    class PV64Stats {
    public:
      PV64Stats(Module64&);
      ~PV64Stats();
    public:
      void allocate(const std::string& title);
      void update();
    private:
      Module64& _m;
      std::vector<Pgp*> _pgp;
      std::vector<Pds_Epics::EpicsPVA*> _pv;
      class PVCache {
      public:
        unsigned value[4];
      };
      std::vector<PVCache> _v;
    };
  };
};

#endif
