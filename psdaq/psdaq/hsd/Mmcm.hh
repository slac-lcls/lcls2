#ifndef HSD_Mmcm_hh
#define HSD_Mmcm_hh

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
      vuint32_t rsvd0[6];
      vuint32_t ClkOut5_1;
      vuint32_t ClkOut5_2;
      vuint32_t ClkOut0_1;
      vuint32_t ClkOut0_2;
      vuint32_t ClkOut1_1;
      vuint32_t ClkOut1_2;
      vuint32_t ClkOut2_1;
      vuint32_t ClkOut2_2;
      vuint32_t ClkOut3_1;
      vuint32_t ClkOut3_2;
      vuint32_t ClkOut4_1;
      vuint32_t ClkOut4_2;
      vuint32_t ClkOut6_1;
      vuint32_t ClkOut6_2;
      vuint32_t ClkFbOut_1;
      vuint32_t ClkFbOut_2;
      vuint32_t DivClk;
      vuint32_t rsvd1;
      vuint32_t Lock_1;
      vuint32_t Lock_2;
      vuint32_t Lock_3;
      vuint32_t rsvd1B[12];
      vuint32_t PowerU;
      vuint32_t rsvd3[38];
      vuint32_t Filt_1;
      vuint32_t Filt_2;
      vuint32_t rsvd4[0x200-0x50];
    };
  };
};

#endif
