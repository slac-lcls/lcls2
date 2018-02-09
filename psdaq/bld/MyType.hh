#ifndef MyType_hh
#define MyType_hh

#include <vector>
#include <stdint.h>

namespace Bld {
  class MyType {
  public:
    enum { Src = 0xface, Port=11001, IP=0xefff8001 };
    MyType(std::vector<unsigned> channels,
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

#endif
