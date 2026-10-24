/**
 * @file
 * @brief HrEncoderTebData, the TEB input record of the high-rate encoder.
 */
#ifndef Pds_Trg_HrEncoderTebData_hh
#define Pds_Trg_HrEncoderTebData_hh

namespace Pds {
  namespace Trg {

    /** TEB input record made by hrEncoderTebPrimitive.cc from the start of the encoder data payload. */
    struct HrEncoderTebData
    {
        /** Copy the 32-bit position from payload bytes 0-3 and the counters from bytes 4, 5 and 6; m_reserved is not set. */
        HrEncoderTebData(uint8_t* payload) {
            m_position = *(int32_t*)payload;
            m_encErrCnt     = payload[4];
            m_missedTrigCnt = payload[5];
            m_latches       = payload[6];
        };
        int32_t m_position;  ///< Encoder position (payload bytes 0-3).
        uint8_t m_encErrCnt;  ///< Encoder error count (payload byte 4).
        uint8_t m_missedTrigCnt;  ///< Missed trigger count (payload byte 5).
        /** Latch byte (payload byte 6); the code comment says the upper 3 bits are the 3 latch bits. */
        uint8_t m_latches; // upper 3 bits define 3 latch bits
        /** Not set by the constructor; the code comment says it is actually 13 bits. */
        uint8_t m_reserved; // actually 13 bits
    };
  };
};

#endif
