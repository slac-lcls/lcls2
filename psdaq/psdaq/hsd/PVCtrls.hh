#ifndef Hsd_PVCtrls_hh
#define Hsd_PVCtrls_hh

#include <string>
#include <vector>

namespace Pds_Epics { class EpicsPVA; };

namespace Pds {
  class Task;
  namespace HSD {

    class Module;
    class StatePV;

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
      void _setState(Action);
    private:
      std::vector<Pds_Epics::EpicsPVA*> _pv;
      StatePV* _state_pv;
      Module& _m;
      Pds::Task& _task;
    };
  };
};

#endif
