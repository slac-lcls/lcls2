/**
 * @file
 * @brief PVStats, the module-wide EPICS statistics PVs of an XPM (timing link counters, PLL status, XTPG time stamps and per-link status).
 */
#ifndef Xpm_PVStats_hh
#define Xpm_PVStats_hh

#include <string>
#include <vector>

#include "psdaq/xpm/Module.hh"

namespace Pds_Epics {
  class PVCached;
};

namespace Pds {
  class Semaphore;
  namespace Xpm {

    /** Module-wide EPICS statistics PVs of an XPM; update() publishes rates computed from the change of the Module counters since the previous call. */
    class PVStats {
    public:
      /** Store the module and semaphore and take the current time, Module::counts() and Module::linkStatus() as the first reference sample. */
      PVStats(Module&, Semaphore&);
      /** Does nothing; the PVs are not deleted. */
      ~PVStats();
    public:
      /** Delete earlier PVs and create, under title: the timing counter PVs for Us and Cu, the PLL PVs of each AMC, the XTPG TimeStamp, PulseId, FiducialIntv and FiducialErr PVs, eight status PVs per link for 32 links, and RecClk, FbClk and BpClk. */
      void allocate(const std::string& title);
      /** Sample the time, Module counters, link status, PLL status and the three monitor clocks, publish them with the overload below while holding the semaphore, and keep the samples for the next call. CPSW exceptions from the publishing step are caught and printed. */
      void update();
      /** Publish the PVs in the order allocate() created them: timing counter rates (clock counts scaled by 16e-6), PLL status, the XTPG time stamp, pulse ID and fiducial interval and error bit, per-link status and count changes, and the clocks scaled by 1e-6. Each value is written only if its PV is connected. */
      void update(const CoreCounts& nc, const CoreCounts& oc, 
                  const LinkStatus* nl, const LinkStatus* ol,
                  const PllStats* pll,
                  unsigned recClk,
                  unsigned fbClk,
                  unsigned bpClk,
                  double dt);
    private:
      void _allocTiming (const std::string&, const char*);
      void _allocPll    (const std::string&, unsigned);
      void _updateTiming(const TimingCounts& nc, 
                         const TimingCounts& oc,
                         double dt,
                         std::vector<Pds_Epics::PVCached*>::iterator&);
      void _updatePll   (const PllStats&,
                         std::vector<Pds_Epics::PVCached*>::iterator&);
    private:
      Module&    _dev;
      Semaphore& _sem;
      std::vector<Pds_Epics::PVCached*> _pv;
      timespec   _t;
      CoreCounts _c;
      LinkStatus _links[32];
    };
  };
};

#endif
