#ifndef HSD_Adt7411_hh
#define HSD_Adt7411_hh

#include "Globals.hh"
#include "psdaq/mmhw/RegProxy.hh"

namespace Pds {
  namespace HSD {
    class Adt7411_Mon {
    public:
      double Tint;  // degC
      double Text;  // degC
      double Vdd;   // Volt
      double ain[8];// Volt
    };

    class Adt7411 {
    public:
      unsigned deviceId       () const;
      unsigned manufacturerId () const;
      unsigned interruptStatus() const;
      unsigned interruptMask  () const;
      unsigned internalTemp   () const;
      unsigned externalTemp   () const;
      void        start       ();
      Adt7411_Mon mon         ();
      void     dump           ();
    public:
      void     interruptMask  (unsigned);
    private:
      Pds::Mmhw::RegProxy _reg[256];
    };
  };
};

#endif
