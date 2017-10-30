#ifndef HSD_Mmcm_hh
#define HSD_Mmcm_hh

#include <stdint.h>

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
      uint32_t rsvd0[6];
      uint32_t ClkOut5_1;
      uint32_t ClkOut5_2;
      uint32_t ClkOut0_1;
      uint32_t ClkOut0_2;
      uint32_t ClkOut1_1;
      uint32_t ClkOut1_2;
      uint32_t ClkOut2_1;
      uint32_t ClkOut2_2;
      uint32_t ClkOut3_1;
      uint32_t ClkOut3_2;
      uint32_t ClkOut4_1;
      uint32_t ClkOut4_2;
      uint32_t ClkOut6_1;
      uint32_t ClkOut6_2;
      uint32_t ClkFbOut_1;
      uint32_t ClkFbOut_2;
      uint32_t DivClk;
      uint32_t rsvd1;
      uint32_t Lock_1;
      uint32_t Lock_2;
      uint32_t Lock_3;
      uint32_t rsvd1B[12];
      uint32_t PowerU;
      uint32_t rsvd3[38];
      uint32_t Filt_1;
      uint32_t Filt_2;
      uint32_t rsvd4[0x200-0x50];
    };
  };
};

#endif
