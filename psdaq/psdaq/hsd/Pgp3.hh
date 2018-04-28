#ifndef HSD_Pgp3_hh
#define HSD_Pgp3_hh

#include "psdaq/hsd/Pgp.hh"

namespace Pds {
  namespace Mmhw { class Pgp3Axil; }
  namespace HSD {
    class Pgp3 : public Pgp {
    public:
      Pgp3(Mmhw::Pgp3Axil&);
    public:
      virtual bool   localLinkReady () const;
      virtual bool   remoteLinkReady() const;
      virtual double   txClkFreqMHz () const;
      virtual double   rxClkFreqMHz () const;
      virtual unsigned txCount      () const;
      virtual unsigned txErrCount   () const;
      virtual unsigned rxOpCodeCount() const;
      virtual unsigned rxOpCodeLast () const;
    private:
      Mmhw::Pgp3Axil& _axi;
    };
  };
};

#endif
