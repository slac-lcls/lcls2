#ifndef Hsd_PV126Stats_hh
#define Hsd_PV126Stats_hh

#include "PvDef.hh"
#include <string>
#include <vector>

namespace Pds_Epics {
  class EpicsPVA;
};

namespace Pds {
  namespace HSD {
    class Module126;
    class Pgp;

    class PV126Stats {
    public:
      PV126Stats(Module126&);
      ~PV126Stats();
    public:
      void allocate(const std::string& title);
      void update();
    private:
      Module126& _m;
      std::vector<Pgp*> _pgp;
      std::vector<Pds_Epics::EpicsPVA*> _pv;

      MonTiming     _v_monTiming[1];
      MonPgp        _v_monPgp   [1];
      MonBuf        _v_monRawBuf[1];
      MonBuf        _v_monFexBuf[1];
      MonBufDetail  _v_monRawDet[1];
      MonEnv        _v_monEnv   [1];
      MonAdc        _v_monAdc   [1];
      MonJesd       _v_monJesd  [1];

      MonTimingCalc _p_monTiming[1];

      class PVCache {
      public:
        unsigned value[4];
      };
      std::vector<PVCache> _v;
    };
  };
};

#endif
