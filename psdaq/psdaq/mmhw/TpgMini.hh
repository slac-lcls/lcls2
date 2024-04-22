#ifndef Pds_Mmhw_TprCore_hh
#define Pds_Mmhw_TprCore_hh

#include "psdaq/mmhw/Reg.hh"
#include <stdint.h>

namespace Pds {
  namespace Mmhw {
    class TprCore {
    public:
      TprCore() {}
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
      Reg       SOFcounts;
      Reg       EOFcounts;
      Reg       Msgcounts;
      Reg       CRCerrors;
      Reg       RxRecClks;
      Reg       RxRstDone;
      Reg       RxDecErrs;
      Reg       RxDspErrs;
      Reg       CSR;
      uint32_t  reserved;
      Reg       TxRefClks;
      Reg       BypassCnts;
      Reg       Version;
    };
  };
};

#endif
