#ifndef Hsd_PVCtrls_hh
#define Hsd_PVCtrls_hh

#include <string>
#include <vector>

namespace Pds_Epics { class EpicsCA; };

namespace Pds {
  namespace HSD {

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
    public:
      void configure  ();
      void unconfigure();
      void reset    ();
      void loopback (bool);
    private:
      std::vector<Pds_Epics::EpicsCA*> _pv;
      Module& _m;
    };
  };
};

#endif
