#ifndef Pds_HSD_TprCore_hh
#define Pds_HSD_TprCore_hh

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
      vuint32_t SOFcounts;
      vuint32_t EOFcounts;
      vuint32_t Msgcounts;
      vuint32_t CRCerrors;
      vuint32_t RxRecClks;
      vuint32_t RxRstDone;
      vuint32_t RxDecErrs;
      vuint32_t RxDspErrs;
      vuint32_t CSR;
      uint32_t  reserved;
      vuint32_t TxRefClks;
      vuint32_t BypassCnts;
      vuint32_t Version;
    };
  };
};

#endif
