#ifndef Pds_Eb_CubeResultDgram_hh
#define Pds_Eb_CubeResultDgram_hh

#include "ResultDgram.hh"

#define ADDBITS(FLD,VAL) auxdata((auxdata()&~s_##FLD) | ((VAL&m_##FLD)<<v_##FLD))
#define GETBITS(FLD)     (auxdata()&s_##FLD)>>v_##FLD

namespace Pds {
    namespace Eb {

        class CubeResultDgram : public ResultDgram
        {
            enum { v_record_evt  =  0, k_record_evt  =  1 };
            enum { v_record_bin  =  1, k_record_bin  =  1 };
            enum { v_monitor_bin =  2, k_monitor_bin =  1 };
            enum { v_flush       =  3, k_flush       =  1 };
            enum { v_bin         = 12, k_bin         = 20 };

            enum { m_bin         = ((1 << k_bin        ) - 1), s_bin         = (m_bin         << v_bin        ) };
            enum { m_record_evt  = ((1 << k_record_evt ) - 1), s_record_evt  = (m_record_evt  << v_record_evt ) };
            enum { m_record_bin  = ((1 << k_record_bin ) - 1), s_record_bin  = (m_record_bin  << v_record_bin ) };
            enum { m_monitor_bin = ((1 << k_monitor_bin) - 1), s_monitor_bin = (m_monitor_bin << v_monitor_bin) };
            enum { m_flush       = ((1 << k_flush      ) - 1), s_flush       = (m_flush       << v_flush      ) };
        public:
            CubeResultDgram(const Pds::EbDgram& dgram, unsigned id) :
                ResultDgram(dgram, id)
            {}
        public:
            void     record       (bool     value) { ADDBITS(record_evt,(value?1:0)); }
            void     binIndex     (uint32_t value) { ADDBITS(bin,value); }
            void     updateRecord (bool     value) { ADDBITS(record_bin,(value?1:0)); }
            void     updateMonitor(bool     value) { ADDBITS(monitor_bin,(value?1:0)); }
            void     flush        (bool     value) { ADDBITS(flush,(value?1:0)); }
            uint32_t binIndex     () const { return GETBITS(bin); }
            bool     record       () const { return GETBITS(record_evt); }
            bool     updateRecord () const { return GETBITS(record_bin); }
            bool     updateMonitor() const { return GETBITS(monitor_bin); }
            bool     flush        () const { return GETBITS(flush); }
        };
    };
};

#undef ADDBITS
#undef GETBITS

#endif
