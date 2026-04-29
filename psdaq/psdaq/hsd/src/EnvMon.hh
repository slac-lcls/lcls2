#ifndef EnvMon_hh
#define EnvMon_hh

namespace Pds {
  namespace HSD {
    class EnvMon {
    public:
      double local12v;
      double edge12v;
      double aux12v;
      double fmc12v;
      double local3_3v;
      double local2_5v;
      double local1_8v;
      double totalPower;
      double fmcPower;
      double boardTemp;
    };
  };
};

#endif
