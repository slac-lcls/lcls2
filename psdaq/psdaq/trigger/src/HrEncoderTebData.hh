#ifndef Pds_Trg_HrEncoderTebData_hh
#define Pds_Trg_HrEncoderTebData_hh

namespace Pds {
  namespace Trg {

    struct HrEncoderTebData
    {
        HrEncoderTebData(uint8_t* payload) {
            m_position = *(int32_t*)payload;
            m_encErrCnt     = payload[4];
            m_missedTrigCnt = payload[5];
            m_latches       = payload[6];
        };
        int32_t m_position;
        uint8_t m_encErrCnt;
        uint8_t m_missedTrigCnt;
        uint8_t m_latches; // upper 3 bits define 3 latch bits
        uint8_t m_reserved; // actually 13 bits
    };
  };
};

#endif
