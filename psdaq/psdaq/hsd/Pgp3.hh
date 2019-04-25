#ifndef HSD_Pgp3_hh
#define HSD_Pgp3_hh

#include "psdaq/hsd/Pgp.hh"

namespace Pds {
  namespace Mmhw { class Pgp3AxilBase; }
  namespace HSD {
    class Pgp3 : public Pgp {
    public:
      Pgp3(Mmhw::Pgp3AxilBase&);
    public:
      virtual void   resetCounts    ();
      virtual void   loopback       (bool v);
      virtual void   skip_interval  (unsigned v);
    public:
      virtual bool   localLinkReady () const;
      virtual bool   remoteLinkReady() const;
      virtual double   txClkFreqMHz () const;
      virtual double   rxClkFreqMHz () const;
      virtual unsigned txCount      () const;
      virtual unsigned txErrCount   () const;
      virtual unsigned rxOpCodeCount() const;
      virtual unsigned rxOpCodeLast () const;
      virtual unsigned remPause     () const;
      virtual bool     loopback     () const;
    private:
      Mmhw::Pgp3AxilBase& _axi;
    };
  };
};

#endif
