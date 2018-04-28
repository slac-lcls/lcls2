#ifndef HSD_Pgp_hh
#define HSD_Pgp_hh

namespace Pds {
  namespace HSD {
    class Pgp {
    public:
      virtual ~Pgp() {}
    public:
      virtual bool   localLinkReady () const = 0;
      virtual bool   remoteLinkReady() const = 0;
      virtual double   txClkFreqMHz () const = 0;
      virtual double   rxClkFreqMHz () const = 0;
      virtual unsigned txCount      () const = 0;
      virtual unsigned txErrCount   () const = 0;
      virtual unsigned rxOpCodeCount() const = 0;
      virtual unsigned rxOpCodeLast () const = 0;
    };
  };
};

#endif
