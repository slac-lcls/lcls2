#include "psdaq/xpm/PVLinkCtrls.hh"
#include "psdaq/xpm/Module.hh"

#include <sstream>

#include <stdio.h>

using Pds_Epics::EpicsPVA;
using Pds_Epics::PVMonitorCb;

namespace Pds {
  namespace Xpm {

    PVLinkCtrls::PVLinkCtrls(Module& m) : _pv(0), _m(m) {}
    PVLinkCtrls::~PVLinkCtrls() {}

    void PVLinkCtrls::allocate(const std::string& title)
    {
    }

    Module& PVLinkCtrls::module() { return _m; }
  };
};
