#ifndef Pds_Eb_WindowResultDgram_hh
#define Pds_Eb_WindowResultDgram_hh

#include "ResultDgram.hh"
#include <stdio.h>

namespace Pds {
    namespace Eb {

        class WindowResultDgram : public ResultDgram
        {
            enum { v_add_bin     =  0, k_add_bin     =  6 }; 
            enum { v_record_bin  =  6, k_record_bin  =  6 };
            enum { v_monitor_bin = 12, k_monitor_bin =  6 };
            enum { v_flush_bin   = 18, k_flush_bin   =  6 };

            enum { m_add_bin     = ((1 << k_add_bin    ) - 1), s_add_bin     = (m_add_bin     << v_add_bin    ) };
            enum { m_record_bin  = ((1 << k_record_bin ) - 1), s_record_bin  = (m_record_bin  << v_record_bin ) };
            enum { m_monitor_bin = ((1 << k_monitor_bin) - 1), s_monitor_bin = (m_monitor_bin << v_monitor_bin) };
            enum { m_flush_bin   = ((1 << k_flush_bin  ) - 1), s_flush_bin   = (m_flush_bin   << v_flush_bin  ) };
        public:
            WindowResultDgram(const Pds::EbDgram& dgram, unsigned id) :
                ResultDgram(dgram, id) {}
        public:
            void     updateAdd    (unsigned value) { ADDBITS(add_bin    ,value); }
            void     updateRecord (unsigned value) { ADDBITS(record_bin ,value); }
            void     updateMonitor(unsigned value) { ADDBITS(monitor_bin,value); }
            void     flush        (unsigned value) { ADDBITS(flush_bin  ,value); }
            uint32_t updateAdd    () const { return GETBITS(add_bin); }
            uint32_t updateRecord () const { return GETBITS(record_bin); }
            uint32_t updateMonitor() const { return GETBITS(monitor_bin); }
            uint32_t flush        () const { return GETBITS(flush_bin); }
            void     dump         (const char* title) const {
                printf("%s: WindowResult",title);
                const uint32_t* p = (const uint32_t*)this;
                for(unsigned i=0; i<sizeof(*this)/4; i++)
                    printf(" %08x", p[i]);
                printf("\n");
            }
        };
    };
};

#undef ADDBITS
#undef GETBITS

#endif
