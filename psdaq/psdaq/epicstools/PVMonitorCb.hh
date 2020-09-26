#ifndef Pds_PVMonitorCb_hh
#define Pds_PVMonitorCb_hh

namespace Pds_Epics {
  class PVMonitorCb {
  public:
    virtual ~PVMonitorCb() {}
    virtual void onConnect() {};
    virtual void onDisconnect() {};
    virtual void updated() = 0;
  };
};

#endif
