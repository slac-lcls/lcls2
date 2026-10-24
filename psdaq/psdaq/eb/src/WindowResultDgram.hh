/**
 * @file
 * @brief WindowResultDgram, a ResultDgram whose auxdata field carries four 6-bit window bin fields. Undefines ADDBITS and GETBITS at its end.
 */
#ifndef Pds_Eb_WindowResultDgram_hh
#define Pds_Eb_WindowResultDgram_hh

#include "ResultDgram.hh"
#include <stdio.h>

namespace Pds {
    namespace Eb {

        /** ResultDgram whose auxdata field holds four 6-bit fields: add (bits 5-0), record (bits 11-6), monitor (bits 17-12) and flush (bits 23-18). */
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
            /** Construct as ResultDgram(dgram, id). */
            WindowResultDgram(const Pds::EbDgram& dgram, unsigned id) :
                ResultDgram(dgram, id) {}
        public:
            /** Set the add field (auxdata bits 5-0) to value, masked to 6 bits. */
            void     updateAdd    (unsigned value) { ADDBITS(add_bin    ,value); }
            /** Set the record field (auxdata bits 11-6) to value, masked to 6 bits. */
            void     updateRecord (unsigned value) { ADDBITS(record_bin ,value); }
            /** Set the monitor field (auxdata bits 17-12) to value, masked to 6 bits. */
            void     updateMonitor(unsigned value) { ADDBITS(monitor_bin,value); }
            /** Set the flush field (auxdata bits 23-18) to value, masked to 6 bits. */
            void     flush        (unsigned value) { ADDBITS(flush_bin  ,value); }
            /** Return the add field (auxdata bits 5-0). */
            uint32_t updateAdd    () const { return GETBITS(add_bin); }
            /** Return the record field (auxdata bits 11-6). */
            uint32_t updateRecord () const { return GETBITS(record_bin); }
            /** Return the monitor field (auxdata bits 17-12). */
            uint32_t updateMonitor() const { return GETBITS(monitor_bin); }
            /** Return the flush field (auxdata bits 23-18). */
            uint32_t flush        () const { return GETBITS(flush_bin); }
            /** Print title followed by the whole object as 32-bit hex words to stdout. */
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
