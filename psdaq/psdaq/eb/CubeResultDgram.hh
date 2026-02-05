#ifndef Pds_Eb_CubeResultDgram_hh
#define Pds_Eb_CubeResultDgram_hh

#include "ResultDgram.hh"

#define ADDBITS(FLD,VAL) auxdata((auxdata()&~s_##FLD) | ((VAL&m_##FLD)<<v_##FLD))
#define GETBITS(FLD)     (auxdata()&s_##FLD)>>v_##FLD

namespace Pds {
    namespace Eb {

        class CubeResultDgram : public ResultDgram
        {
            enum { v_bin     =  0, k_bin     = 10 };
            enum { v_record  = 10, k_record  =  1 };
            enum { v_monitor = 11, k_monitor =  1 };

            enum { m_bin     = ((1 << k_bin    ) - 1), s_bin     = (m_bin     << v_bin    ) };
            enum { m_record  = ((1 << k_record ) - 1), s_record  = (m_record  << v_record ) };
            enum { m_monitor = ((1 << k_monitor) - 1), s_monitor = (m_monitor << v_monitor) };
        public:
            CubeResultDgram(const Pds::EbDgram& dgram, unsigned id) :
                ResultDgram(dgram, id)
            {}
        public:
            void     binIndex     (uint32_t value) { ADDBITS(bin,value); }
            void     updateRecord (bool     value) { ADDBITS(record,(value?1:0)); }
            void     updateMonitor(bool     value) { ADDBITS(monitor,(value?1:0)); }
            uint32_t binIndex     () const { return GETBITS(bin); }
            bool     updateRecord () const { return GETBITS(record); }
            bool     updateMonitor() const { return GETBITS(monitor); }
        };
    };
};

#undef ADDBITS
#undef GETBITS

#endif
