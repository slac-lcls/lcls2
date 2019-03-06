#ifndef Xpm_PVLinkCtrls_hh
#define Xpm_PVLinkCtrls_hh

#include "psdaq/epicstools/EpicsPVA.hh"

#include <string>
#include <vector>

namespace Pds {
  namespace Xpm {

    class Module;

    class PVLinkCtrls
    {
    public:
      PVLinkCtrls(Module&);
      ~PVLinkCtrls();
    public:
      void allocate(const std::string& title);
      void update();
    public:
      Module& module();
    private:
      std::vector<Pds_Epics::EpicsPVA*> _pv;
      Module&  _m;
    };
  };
};

#endif
