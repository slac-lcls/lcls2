/**
 * @file
 * @brief TriggerData_xpphsd, the TEB input record of the example XPP digitizer trigger primitive.
 */
#ifndef Pds_Trg_TriggerData_xpphsd_hh
#define Pds_Trg_TriggerData_xpphsd_hh

namespace Pds {
  namespace Trg {

    /** TEB input record written by TriggerPrimitiveExample_xpphsd.cc. */
    struct TriggerData_xpphsd
    {
      /** Store nPeaks_. */
      TriggerData_xpphsd(uint64_t nPeaks_) : nPeaks(nPeaks_) {};
      uint64_t nPeaks;  ///< Peak count supplied by the example primitive.
    };
  };
};

#endif
