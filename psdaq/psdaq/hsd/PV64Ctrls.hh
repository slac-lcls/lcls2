#ifndef Hsd_PV64Ctrls_hh
#define Hsd_PV64Ctrls_hh

#include <string>
#include <vector>

namespace Pds_Epics { class EpicsPVA; };

namespace Pds {
  class Task;
  namespace HSD {

    class Module64;

    enum Action { Configure, Unconfigure, Reset, ConfigureA, ConfigureB };
    enum State  { InTransition=0, Configured=1, Unconfigured=2 };

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
      void configure ();
      void configure (unsigned);
      void enable    ();
      void disable   ();
      void reset     ();
      void loopback  (bool);
    public:
      static void interleave(bool);
    private:
      void _setState(State);
    private:
      std::vector<Pds_Epics::EpicsPVA*> _pv;
      Pds_Epics::EpicsPVA* _state_pv;
      Pds_Epics::EpicsPVA* _readyA;
      Pds_Epics::EpicsPVA* _readyB;
      Module64& _m;
      Pds::Task& _task;
    };
  };
};

#endif
