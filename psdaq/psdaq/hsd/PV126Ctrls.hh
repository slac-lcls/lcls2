#ifndef Hsd_PV126Ctrls_hh
#define Hsd_PV126Ctrls_hh

#include "PVCtrlsBase.hh"
#include <string>
#include <vector>

namespace Pds_Epics { class EpicsPVA; };

namespace Pds {
  namespace HSD {
    class Module126;
    class PV126Ctrls : public PVCtrlsBase
    {
    public:
      PV126Ctrls(Module126&, Pds::Task&);
      ~PV126Ctrls() {}
    public:
      Module126& module() { return _m; }
    public:
      void reset     ();
      void configure (unsigned);
      void loopback  (bool);
    private:
      void _allocate();
      int  _nchips  () { return 1; }
    private:
      Module126& _m;
    };
  };
};

#endif
