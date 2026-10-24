/**
 * @file
 * @brief PVPStats, the EPICS statistics PVs (L0 rates, counts and dead time) of one XPM partition.
 */
#ifndef Xpm_PVPStats_hh
#define Xpm_PVPStats_hh

#include <string>
#include <vector>

#include "psdaq/xpm/Module.hh"

namespace Pds_Epics {
  class PVCached;
};

namespace Pds {
  namespace Xpm {
    /** EPICS statistics PVs of one XPM partition, updated from Module::l0Stats() by update(). */
    class PVPStats {
    public:
      /** Store the module and partition; no PVs until allocate(). */
      PVPStats(Module&, unsigned partition);
      /** Does nothing; the PVs are not deleted. */
      ~PVPStats();
    public:
      /** Delete earlier PVs and create the partition PVs named title, a colon and L0InpRate, L0AccRate, L1Rate, NumL0Inp, NumL0Acc, NumL1, DeadFrac, DeadTime, RunTime or MsgDelay, plus the 32-element dttitle:DeadFLnk. */
      void allocate(const std::string& title,
                    const std::string& dttitle);
      /** Select the partition, read Module::l0Stats() and publish values computed from the change since the last call; the fiducial period is taken as 14/13 us. If enabled, write RunTime, MsgDelay (the L0 delay), the L0 input and accept rates, the counts, DeadFrac and, if L0 was enabled in the interval, DeadTime and the per-link DeadFLnk. Otherwise write only DeadFLnk, from the per-link inhibit times over the elapsed wall-clock time; L1Rate and NumL1 are never written. */
      void update(bool);
    private:
      Module&                           _dev;
      unsigned                          _partition;
      std::vector<Pds_Epics::PVCached*> _pv;
      L0Stats                           _begin;
      L0Stats                           _last;
    };
  };
};

#endif
