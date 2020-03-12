#ifndef Hsd_PVCtrlsBase_hh
#define Hsd_PVCtrlsBase_hh

#include <string>
#include <vector>

namespace Pds_Epics { class EpicsPVA; };

namespace Pds {
  class Task;
  namespace HSD {

    enum Action { Unconfigure, ResetA, ResetB, ConfigureA, ConfigureB };
    class FexCfg;

    class PVCtrlsBase
    {
    public:
      PVCtrlsBase(Pds::Task&);
      virtual ~PVCtrlsBase();
    public:
      void allocate(const std::string& title);
      void call    (Action);
    public:
      virtual void reset     (unsigned) = 0;
      virtual void configure (unsigned) = 0;
      virtual void loopback  (bool) = 0;
    private:
      virtual void _allocate () = 0;
      virtual int  _nchips   () = 0;
    protected:
      void _configure_fex (unsigned,FexCfg&);
    protected:
      std::vector<Pds_Epics::EpicsPVA*> _pv;
      Pds_Epics::EpicsPVA* _ready[2];
      Pds::Task& _task;
    };
  };
};

#endif
