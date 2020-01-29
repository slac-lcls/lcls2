#ifndef Hsd_PV134Stats_hh
#define Hsd_PV134Stats_hh

#include "PvDef.hh"
#include <string>
#include <vector>

namespace Pds_Epics {
  class EpicsPVA;
};

namespace Pds {
  namespace HSD {
    class Pgp;
    class Module134;

    class PV134Stats {
    public:
      PV134Stats(Module134&);
      ~PV134Stats();
    public:
      void allocate(const std::string& title);
      void update();
    private:
      Module134& _m;
      std::vector<Pgp*> _pgp;
      std::vector<Pds_Epics::EpicsPVA*> _pv;

      MonTiming     _v_monTiming[2];
      MonPgp        _v_monPgp   [2];
      MonBuf        _v_monRawBuf[2];
      MonBuf        _v_monFexBuf[2];
      MonBufDetail  _v_monRawDet[2];
      MonFlow       _v_monFlow  [2];
      MonEnv        _v_monEnv   [2];
      MonAdc        _v_monAdc   [2];
      MonJesd       _v_monJesd  [2];

      MonTimingCalc _p_monTiming[2];

      class PVCache {
      public:
        unsigned value[4];
      };
      std::vector<PVCache> _v;
    };
  };
};

#endif
