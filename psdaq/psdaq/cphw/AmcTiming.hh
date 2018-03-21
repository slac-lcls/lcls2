#ifndef Pds_Cphw_AmcTiming_hh
#define Pds_Cphw_AmcTiming_hh

#include "psdaq/cphw/Reg.hh"
#include "psdaq/cphw/AxiVersion.hh"
#include "psdaq/cphw/RingBuffer.hh"
#include "psdaq/cphw/XBar.hh"

namespace Pds {
  namespace Cphw {
    class AmcTiming {
    public:
      void setPolarity     (bool);
      void setLCLS         ();
      void setLCLSII       ();
      void resetStats      ();
      void bbReset         ();
      void dumpStats       () const;
      void setRxAlignTarget(unsigned);
      void setRxResetLength(unsigned);
      void dumpRxAlign     () const;
      bool linkUp          () const;
    public:
      //  AxiVersion @ 0
      AxiVersion version;
      uint32_t rsvd_version[(0x03000000-sizeof(AxiVersion))>>2];
      //  AxiSy56040 @ 0x03000000
      XBar       xbar;
      uint32_t rsvd_xbar[(0x05000000-sizeof(XBar))>>2];
      //  TimingRx   @ 0x08000000
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
      //  RingBuffer @ 0x08010000
      RingBuffer ring0;
      uint32_t rsvd_ring0[(0x10000-sizeof(RingBuffer))>>2];
      //  RingBuffer @ 0x08020000
      RingBuffer ring1;
      uint32_t rsvd_ring1[(0x10000-sizeof(RingBuffer))>>2];
      //  TPGMini    @ 0x08030000
      uint32_t rsvd_xx[(0x800000-0x30000)>>2];
      //  GthRxAlign @ 0x08800000
      Reg gthAlign[64];
      Reg gthAlignTarget;
      Reg gthAlignLast;
    };
  };
};

#endif

