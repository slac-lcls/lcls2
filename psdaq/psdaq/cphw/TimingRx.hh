#ifndef Pds_Cphw_TimingRx_hh
#define Pds_Cphw_TimingRx_hh

#include "psdaq/cphw/Reg.hh"

namespace Pds {
  namespace Cphw {
    class TimingRx {
    public:
      void setPolarity     (bool);
      void setLCLS         ();
      void setLCLSII       ();
      void resetStats      ();
      void bbReset         ();
      void dumpStats       () const;
      bool linkUp          () const;
    public:
      Reg SOFcounts;
      Reg EOFcounts;
      Reg Msgcounts;
      Reg CRCerrors;
      Reg RxRecClks;
      Reg RxRstDone;
      Reg RxDecErrs;
      Reg RxDspErrs;
      Reg CSR;
      Reg MsgDelay;
      Reg TxRefClks;
      Reg BuffByCnts;
      uint32_t rsvd14[(0x10000>>2)-12];
    };
  };
};

#endif

