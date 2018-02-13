#ifndef Pds_Cphw_AmcPLL_hh
#define Pds_Cphw_AmcPLL_hh

#include "psdaq/cphw/Reg.hh"

namespace Pds {
  namespace Cphw {
    class AmcPLL {
    public:
      void BwSel    (unsigned);
      void FrqTbl   (unsigned);
      void FrqSel   (unsigned);
      void RateSel  (unsigned);
      void PhsInc   ();
      void PhsDec   ();
      void Bypass   (bool);
      void Reset    ();
      unsigned BwSel  () const;
      unsigned FrqTbl () const;
      unsigned FrqSel () const;
      unsigned RateSel() const;
      bool     Bypass () const;
      unsigned Status0() const;
      unsigned Count0 () const;
      unsigned Status1() const;
      unsigned Count1 () const;
      void Skew       (int);
      void dump       () const;
    public:
      //  0x0010 - RW:  configuration for AMC[index]
      //  [3:0]   bwSel         Loop filter bandwidth selection
      //  [5:4]   frqTbl        Frequency table - character {L,H,M} = 00,01,1x
      //  [15:8]  frqSel        Frequency selection - 4 characters
      //  [19:16] rate          Rate - 2 characters
      //  [20]    inc           Increment phase
      //  [21]    dec           Decrement phase
      //  [22]    bypass        Bypass 
      //  [23]    rstn          Reset  (inverted)
      //  [26:24] Count[0]    count[0] for AMC[index] (RO)
      //  [27]    Stat[0]     stat[0]  for AMC[index] (RO)
      //  [30:28] Count[1]    count[1] for AMC[index] (RO)
      //  [31]    Stat[1]     stat[1]  for AMC[index] (RO)
      Cphw::Reg   _config;
    };
  };
};

#endif
