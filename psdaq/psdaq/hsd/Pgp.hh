#ifndef HSD_Pgp_hh
#define HSD_Pgp_hh

namespace Pds {
  namespace HSD {
    class Pgp {
    public:
      virtual ~Pgp() {}
    public:
      virtual void   resetCounts    ()       = 0;
      virtual void   loopback       (bool)   = 0;
      virtual void   skip_interval  (unsigned) = 0;
    public:
      virtual bool   localLinkReady () const = 0;
      virtual bool   remoteLinkReady() const = 0;
      virtual double   txClkFreqMHz () const = 0;
      virtual double   rxClkFreqMHz () const = 0;
      virtual unsigned txCount      () const = 0;
      virtual unsigned txErrCount   () const = 0;
      virtual unsigned rxOpCodeCount() const = 0;
      virtual unsigned rxOpCodeLast () const = 0;
      virtual unsigned remPause     () const = 0;
      virtual bool     loopback     () const = 0;
    };
  };
};

#endif
