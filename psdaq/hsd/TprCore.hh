#ifndef Pds_HSD_TprCore_hh
#define Pds_HSD_TprCore_hh

#include <unistd.h>
#include <stdint.h>

namespace Pds {
  namespace HSD {
    class TprCore {
    public:
      bool rxPolarity () const;
      void rxPolarity (bool p);
      void resetRx    ();
      void resetRxPll ();
      void resetCounts();
      void setLCLS    ();
      void setLCLSII  ();
      void dump() const;
    public:
      volatile uint32_t SOFcounts;
      volatile uint32_t EOFcounts;
      volatile uint32_t Msgcounts;
      volatile uint32_t CRCerrors;
      volatile uint32_t RxRecClks;
      volatile uint32_t RxRstDone;
      volatile uint32_t RxDecErrs;
      volatile uint32_t RxDspErrs;
      volatile uint32_t CSR;
      uint32_t          reserved;
      volatile uint32_t TxRefClks;
      volatile uint32_t BypassCnts;
      volatile uint32_t Version;
    };
  };
};

#endif
