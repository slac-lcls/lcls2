/**
 * @file
 * @brief TimingTebData, the TEB input record of the timing-system trigger primitive.
 */
#ifndef Pds_Trg_TimingTebData_hh
#define Pds_Trg_TimingTebData_hh

namespace Pds {
  namespace Trg {

    /** TEB input record written by timingTebPrimitive.cc from the timing data of an event. */
    struct TimingTebData
    {
        /** Store ebeamDestn_ and copy nine 32-bit words from eventcodes_. */
        TimingTebData(const uint8_t   ebeamDestn_,
                      const uint32_t* eventcodes_) {
            ebeamDestn = ebeamDestn_;
            memcpy(eventcodes,eventcodes_,sizeof(eventcodes));
        };
        uint8_t  ebeamDestn;  ///< Beam destination with bit 7 set when the beam is present (as filled by timingTebPrimitive.cc).
        uint32_t eventcodes[9];  ///< The 18 16-bit sequence words of the timing data, copied as nine 32-bit words.
    };
  };
};

#endif
