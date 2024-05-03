#ifndef HSD_Mmcm_hh
#define HSD_Mmcm_hh

#include "psdaq/mmhw/Reg.hh"
#include "Globals.hh"

namespace Pds {
  namespace HSD {
    class Mmcm {
    public:
      void setLCLS  (unsigned delay_int, unsigned delay_frac);
      void setLCLSII(unsigned delay_int, unsigned delay_frac);
    private:
      void _setFbMult(unsigned);
      void _setClkDiv(unsigned,unsigned);
      void _setLock  (unsigned);
      void _setFilt  (unsigned);
    private:
      Mmhw::Reg rsvd0[6];
      Mmhw::Reg ClkOut5_1;
      Mmhw::Reg ClkOut5_2;
      Mmhw::Reg ClkOut0_1;
      Mmhw::Reg ClkOut0_2;
      Mmhw::Reg ClkOut1_1;
      Mmhw::Reg ClkOut1_2;
      Mmhw::Reg ClkOut2_1;
      Mmhw::Reg ClkOut2_2;
      Mmhw::Reg ClkOut3_1;
      Mmhw::Reg ClkOut3_2;
      Mmhw::Reg ClkOut4_1;
      Mmhw::Reg ClkOut4_2;
      Mmhw::Reg ClkOut6_1;
      Mmhw::Reg ClkOut6_2;
      Mmhw::Reg ClkFbOut_1;
      Mmhw::Reg ClkFbOut_2;
      Mmhw::Reg DivClk;
      Mmhw::Reg rsvd1;
      Mmhw::Reg Lock_1;
      Mmhw::Reg Lock_2;
      Mmhw::Reg Lock_3;
      Mmhw::Reg rsvd1B[12];
      Mmhw::Reg PowerU;
      Mmhw::Reg rsvd3[38];
      Mmhw::Reg Filt_1;
      Mmhw::Reg Filt_2;
      Mmhw::Reg rsvd4[0x200-0x50];
    };
  };
};

#endif
