#ifndef Hsd_PVCtrls_hh
#define Hsd_PVCtrls_hh

#include <string>
#include <vector>

namespace Pds_Epics { class EpicsPVA; };

namespace Pds {
  class Task;
  namespace HSD {

    class Module;

    enum Action { Configure, Unconfigure, Reset };
    enum State  { InTransition=0, Configured=1, Unconfigured=2 };

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
      void configure  ();
      void unconfigure();
      void reset      ();
      void loopback (bool);
    public:
      static void interleave(bool);
    private:
      void _setState(State);
    private:
      std::vector<Pds_Epics::EpicsPVA*> _pv;
      Pds_Epics::EpicsPVA* _state_pv;
      Module& _m;
      Pds::Task& _task;
    };
  };
};

#endif
