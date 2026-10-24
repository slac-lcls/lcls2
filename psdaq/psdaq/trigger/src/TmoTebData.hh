/**
 * @file
 * @brief TmoTebData, the example TEB input record used by the TMO trigger primitive and decision.
 */
#ifndef Pds_Trg_TmoTebData_hh
#define Pds_Trg_TmoTebData_hh

namespace Pds {
  namespace Trg {

    /** TEB input record with a write word and a monitor word; tmoTebPrimitive.cc fills them with fixed example values (0xdeadbeef, 0x12345678) and tmoTeb.cc compares them with its configured values. */
    struct TmoTebData
    {
      /** Store the write and monitor words. */
      TmoTebData(uint32_t write_, uint32_t monitor_) : write(write_), monitor(monitor_) {};
      uint32_t write;  ///< Word compared by tmoTeb.cc with its write value to set the persist bit of the result.
      uint32_t monitor;  ///< Word compared by tmoTeb.cc with its monitor value to request monitoring by all MEBs.
    };
  };
};

#endif
