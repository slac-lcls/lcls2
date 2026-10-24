/**
 * @file
 * @brief PVBase, an EpicsPVA that is its own monitor callback.
 */
#ifndef Pds_PVBase_hh
#define Pds_PVBase_hh

#include "psdaq/epicstools/EpicsPVA.hh"

//
//  Abstract base class for read/write channel access
//
namespace Pds_Epics {
  /** EpicsPVA that is its own monitor callback; the base for read and write PVs (per the code comment). Subclasses override updated() and onConnect(). */
  class PVBase : public EpicsPVA,
                 public PVMonitorCb {
  public:
    /** Connect to channelName through pva with this object as the monitor callback; maxElements is passed on but not used by EpicsPVA. */
    PVBase(const char* channelName, const int maxElements=1) : EpicsPVA(channelName, this, maxElements) {}
    /** Connect to channelName through provider (ca or pva) with this object as the monitor callback. */
    PVBase(const char* provider, const char* channelName, const int maxElements=1) : EpicsPVA(provider, channelName, this, maxElements) {}
    /** Does nothing; empty body. */
    ~PVBase() {}
    /** Does nothing; empty body. */
    void updated() {}
    /** Wait up to 30 s for the first value (getComplete()); returns true if it has arrived. */
    bool ready  () { return getComplete(); }
  };
};

#endif
