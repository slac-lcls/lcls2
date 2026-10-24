/**
 * @file
 * @brief TriggerData_bld, the TEB input record of the example BLD trigger primitive.
 */
#ifndef Pds_Trg_TriggerData_bld_hh
#define Pds_Trg_TriggerData_bld_hh

namespace Pds {
  namespace Trg {

    /** TEB input record of the example BLD trigger (TriggerPrimitiveExample_bld.cc writes it, TriggerExample.cc reads it). */
    struct TriggerData_bld
    {
      /** Store eBeam_. */
      TriggerData_bld(uint64_t eBeam_) : eBeam(eBeam_) {};
      uint64_t eBeam;  ///< Single 64-bit value supplied by the example primitive.
    };
  };
};

#endif
