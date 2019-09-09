#ifndef HSD_FmcCore_hh
#define HSD_FmcCore_hh

#include "Globals.hh"

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
      vuint32_t _irq;
      vuint32_t _irq_en;
      vuint32_t _rsvd[6];
      vuint32_t _detect;
      vuint32_t _cmd;
      vuint32_t _ctrl;
      vuint32_t _rsvd2[5];
      vuint32_t _clock_select;
      vuint32_t _clock_count;
      vuint32_t _rsvd3[0xee];
    };
  };
};

#endif
