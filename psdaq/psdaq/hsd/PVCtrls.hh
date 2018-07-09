#ifndef Hsd_PVCtrls_hh
#define Hsd_PVCtrls_hh

#include <string>
#include <vector>

namespace Pds_Epics { class EpicsCA; };

namespace Pds {
  class Task;
  namespace HSD {

    class Module;

    enum Action { Configure, Unconfigure, EnableTr, DisableTr, Reset };

    class PVCtrls
    {
    public:
      PVCtrls(Module&, Pds::Task&);
      ~PVCtrls();
    public:
      void allocate(const std::string& title);
      void update();
      void call  (Action);
    public:
      Module& module();
    public:
      void configure();
      void enable   ();
      void disable  ();
      void reset    ();
      void loopback (bool);
    public:
      static void interleave(bool);
    private:
      std::vector<Pds_Epics::EpicsCA*> _pv;
      Module& _m;
      Pds::Task& _task;
    };
  };
};

#endif
