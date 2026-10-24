/**
 * @file
 * @brief PVLinkCtrls, a placeholder for per-link EPICS controls of an XPM; allocate() is empty.
 */
#ifndef Xpm_PVLinkCtrls_hh
#define Xpm_PVLinkCtrls_hh

#include "psdaq/epicstools/EpicsPVA.hh"

#include <string>
#include <vector>

namespace Pds {
  namespace Xpm {

    class Module;

    /** Placeholder for per-link EPICS controls: holds a Module but creates no PVs. Not used elsewhere in psdaq. */
    class PVLinkCtrls
    {
    public:
      /** Store the module. */
      PVLinkCtrls(Module&);
      /** Does nothing; empty body. */
      ~PVLinkCtrls();
    public:
      /** Does nothing; empty body. */
      void allocate(const std::string& title);
      /** Declared but not defined in PVLinkCtrls.cc. */
      void update();
    public:
      /** Return the Module. */
      Module& module();
    private:
      std::vector<Pds_Epics::EpicsPVA*> _pv;
      Module&  _m;
    };
  };
};

#endif
