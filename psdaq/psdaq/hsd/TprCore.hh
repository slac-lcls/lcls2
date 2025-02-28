#ifndef Pds_HSD_TprCore_hh
#define Pds_HSD_TprCore_hh

#include "psdaq/mmhw/Reg.hh"
#include "Globals.hh"
#include <stdint.h>

namespace Pds {
  namespace HSD {
    class TprCore {
    public:
      bool rxPolarity () const;
      void rxPolarity (bool p);
      void resetRx    ();
      void resetRxPll ();
      void resetBB    ();
      void resetCounts();
      void setLCLS    ();
      void setLCLSII  ();

      double txRefClockRate() const;
      double rxRecClockRate() const;
      void   dump() const;
    public:
      Mmhw::Reg SOFcounts;
      Mmhw::Reg EOFcounts;
      Mmhw::Reg Msgcounts;
      Mmhw::Reg CRCerrors;
      Mmhw::Reg RxRecClks;
      Mmhw::Reg RxRstDone;
      Mmhw::Reg RxDecErrs;
      Mmhw::Reg RxDspErrs;
      Mmhw::Reg CSR;
      uint32_t  reserved;
      Mmhw::Reg TxRefClks;
      Mmhw::Reg BypassCnts;
      Mmhw::Reg Version;
    };
  };
};

#endif
