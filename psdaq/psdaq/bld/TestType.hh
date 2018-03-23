#ifndef Pds_Bld_TestType_hh
#define Pds_Bld_TestType_hh

#include <vector>
#include <stdint.h>

namespace Pds {
  namespace Bld {
    class TestType {
    public:
      enum { Src = 0xface, Port=11001, IP=0xefff8001 };
      //      enum { Src = 0xface, Port=11001, IP=0x0a000004 };
      TestType(std::vector<unsigned> channels,
               unsigned valid) :
        _damageMask(valid)
      {
      }
    private:
      class BeamVector {
      public:
        float x, y, xp, yp;
      };
      uint32_t   _damageMask;
      float      _bunchCharge;
      float      _dumpCharge;
      float      _beamEnergy;
      float      _photonEnergy;
      float      _pkCurrBC1;
      float      _energyBC1;
      float      _pkCurrBC2;
      float      _energyBC2;
      BeamVector _ltu;
      BeamVector _und;
      BeamVector _launch;
      double     _ltu250x;
      double     _ltu450x;
    };
  };
};

#endif
