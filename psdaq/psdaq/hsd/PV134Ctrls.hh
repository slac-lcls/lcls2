#ifndef Hsd_PV134Ctrls_hh
#define Hsd_PV134Ctrls_hh

#include "PVCtrlsBase.hh"
#include <string>
#include <vector>

namespace Pds {
  namespace HSD {
    class Module134;
    class PV134Ctrls : public PVCtrlsBase
    {
    public:
      PV134Ctrls(Module134&, Pds::Task&);
      ~PV134Ctrls() {}
    public:
      Module134& module() { return _m; }
    public:
      void reset     ();
      void configure (unsigned);
      void loopback  (bool);
    private:
      void _allocate();
      int  _nchips  () { return 2; }
    private:
      Module134& _m;
      unsigned   _testpattern;
    };
  };
};

#endif
