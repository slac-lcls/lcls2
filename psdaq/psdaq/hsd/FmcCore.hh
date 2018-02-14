#ifndef HSD_FmcCore_hh
#define HSD_FmcCore_hh

#include <stdint.h>

namespace Pds {
  namespace HSD {
    class FmcCore {
    public:
      bool   present  () const;
      bool   powerGood() const;
      void   selectClock(unsigned);
      double clockRate() const;
      void   cal_enable ();
      void   cal_disable();
    private:
      uint32_t _irq;
      uint32_t _irq_en;
      uint32_t _rsvd[6];
      uint32_t _detect;
      uint32_t _cmd;
      uint32_t _ctrl;
      uint32_t _rsvd2[5];
      uint32_t _clock_select;
      uint32_t _clock_count;
      uint32_t _rsvd3[0xee];
    };
  };
};

#endif
