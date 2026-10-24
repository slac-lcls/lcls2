/**
 * @file
 * @brief TriggerData_timing, a TEB input record holding nine 32-bit event code words.
 */
#ifndef Pds_Trg_TriggerData_timing_hh
#define Pds_Trg_TriggerData_timing_hh

namespace Pds {
  namespace Trg {

    /** TEB input record holding nine 32-bit event code words; no user was found in psdaq. */
    struct TriggerData_timing
    {
        /** Copy nine 32-bit words from eventcodes_. */
        TriggerData_timing(uint32_t* eventcodes_) { memcpy(eventcodes,eventcodes_,sizeof(eventcodes)); };
        uint32_t eventcodes[9];  ///< Nine 32-bit event code words.
    };
  };
};

#endif
