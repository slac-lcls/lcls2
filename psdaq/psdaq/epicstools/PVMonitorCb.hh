/**
 * @file
 * @brief PVMonitorCb, the callback interface for PV connection and update events.
 */
#ifndef Pds_PVMonitorCb_hh
#define Pds_PVMonitorCb_hh

namespace Pds_Epics {
  /** Callback interface for PV events. EpicsPVA calls updated() for each monitor update; MonTracker also calls onConnect() and onDisconnect(). */
  class PVMonitorCb {
  public:
    /** Does nothing; empty virtual destructor. */
    virtual ~PVMonitorCb() {}
    /** Called by MonTracker on the first data update after connecting; does nothing by default. */
    virtual void onConnect() {};
    /** Called by MonTracker on a disconnect; does nothing by default. */
    virtual void onDisconnect() {};
    /** Pure virtual: called for each update of the PV. */
    virtual void updated() = 0;
  };
};

#endif
