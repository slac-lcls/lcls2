#ifndef Hsd_PV64Ctrls_hh
#define Hsd_PV64Ctrls_hh

#include <string>
#include <vector>

namespace Pds_Epics { class EpicsPVA; };

namespace Pds {
  class Task;
  namespace HSD {

    class Module64;
    class StatePV;

    enum Action { Configure, Unconfigure, EnableTr, DisableTr, Reset };

    class PV64Ctrls
    {
    public:
      PV64Ctrls(Module64&, Pds::Task&);
      ~PV64Ctrls();
    public:
      void allocate(const std::string& title);
      void update();
      void call  (Action);
    public:
      Module64& module();
    public:
      void configure();
      void enable   ();
      void disable  ();
      void reset    ();
      void loopback (bool);
    public:
      static void interleave(bool);
    private:
      void _setState(Action);
    private:
      std::vector<Pds_Epics::EpicsPVA*> _pv;
      StatePV* _state_pv;
      Module64& _m;
      Pds::Task& _task;
    };
  };
};

#endif
