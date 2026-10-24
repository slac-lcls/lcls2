/**
 * @file
 * @brief TriggerData_cam, the TEB input record of the example camera trigger primitive.
 */
#ifndef Pds_Trg_TriggerData_cam_hh
#define Pds_Trg_TriggerData_cam_hh

namespace Pds {
  namespace Trg {

    /** TEB input record written by TriggerPrimitiveExample_cam.cc. */
    struct TriggerData_cam
    {
      /** Store value_. */
      TriggerData_cam(uint64_t value_) : value(value_) {};
      uint64_t value;  ///< Single 64-bit value supplied by the example primitive.
    };
  };
};

#endif
