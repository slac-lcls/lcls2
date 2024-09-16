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
      virtual void   resetCounts    ();
      virtual void   loopback       (bool v);
      virtual void   skip_interval  (unsigned v);
      virtual void   txdiffctrl     (unsigned v);
      virtual void   txprecursor    (unsigned v);
      virtual void   txpostcursor   (unsigned v);
    public:
      virtual bool     localLinkReady () const;
      virtual bool     remoteLinkReady() const;
      virtual unsigned remoteLinkId () const;
      virtual double   txClkFreqMHz () const;
      virtual double   rxClkFreqMHz () const;
      virtual unsigned txCount      () const;
      virtual unsigned txErrCount   () const;
      virtual unsigned rxCount      () const;
      virtual unsigned rxErrCount   () const;
      virtual unsigned rxOpCodeCount() const;
      virtual uint64_t rxOpCodeLast () const;
      virtual unsigned remPause     () const;
      virtual bool     loopback     () const;
    private:
      Mmhw::Pgp3Axil& _axi;
    };
  };
};

#endif
