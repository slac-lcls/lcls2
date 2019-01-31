#ifndef Dti_PVCtrls_hh
#define Dti_PVCtrls_hh

#include "psdaq/epicstools/EpicsCA.hh"

#include <string>
#include <vector>

namespace Pds {
  namespace Dti {

    class Module;

    class PVCtrls
    {
    public:
      PVCtrls(Module&);
      ~PVCtrls();
    public:
      void allocate(const std::string& title);
      void update();
    public:
      Module& module();
    private:
      std::vector<Pds_Epics::EpicsCA*> _pv;
      Module& _m;
    };
  };
};

#endif
