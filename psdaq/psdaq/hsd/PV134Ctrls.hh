#ifndef Hsd_PV134Ctrls_hh
#define Hsd_PV134Ctrls_hh

#include <string>
#include <vector>

namespace Pds_Epics { class EpicsPVA; };

namespace Pds {
  class Task;
  namespace HSD {

    class Module134;

    enum Action { Unconfigure, Reset, ConfigureA, ConfigureB };
    enum State  { InTransition=0, Configured=1, Unconfigured=2 };

    class PV134Ctrls
    {
    public:
      PV134Ctrls(Module134&, Pds::Task&);
      ~PV134Ctrls();
    public:
      void allocate(const std::string& title);
      void update();
      void call  (Action);
    public:
      Module134& module();
    public:
      void configure (unsigned);
      void enable    ();
      void disable   ();
      void reset     ();
      void loopback  (bool);
    private:
      void _setState(State);
    private:
      std::vector<Pds_Epics::EpicsPVA*> _pv;
      Pds_Epics::EpicsPVA* _state_pv;
      Pds_Epics::EpicsPVA* _readyA;
      Pds_Epics::EpicsPVA* _readyB;
      Module134& _m;
      Pds::Task& _task;
      unsigned   _testpattern;
    };
  };
};

#endif
