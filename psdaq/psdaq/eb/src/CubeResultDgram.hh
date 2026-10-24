/**
 * @file
 * @brief CubeResultDgram, a ResultDgram whose auxdata field carries cube binning flags and a bin index.
 */
#ifndef Pds_Eb_CubeResultDgram_hh
#define Pds_Eb_CubeResultDgram_hh

#include "ResultDgram.hh"
#include <stdio.h>

namespace Pds {
    namespace Eb {

        /** ResultDgram whose auxdata field holds a record-event bit (bit 0), a record-bin bit (bit 1), a monitor-bin bit (bit 2), a flush bit (bit 3) and a 20-bit bin index (bits 23-4). */
        class CubeResultDgram : public ResultDgram
        {
            enum { v_record_evt  =  0, k_record_evt  =  1 };
            enum { v_record_bin  =  1, k_record_bin  =  1 };
            enum { v_monitor_bin =  2, k_monitor_bin =  1 };
            enum { v_flush       =  3, k_flush       =  1 };
            enum { v_bin         =  4, k_bin         = 20 };

            enum { m_bin         = ((1 << k_bin        ) - 1), s_bin         = (m_bin         << v_bin        ) };
            enum { m_record_evt  = ((1 << k_record_evt ) - 1), s_record_evt  = (m_record_evt  << v_record_evt ) };
            enum { m_record_bin  = ((1 << k_record_bin ) - 1), s_record_bin  = (m_record_bin  << v_record_bin ) };
            enum { m_monitor_bin = ((1 << k_monitor_bin) - 1), s_monitor_bin = (m_monitor_bin << v_monitor_bin) };
            enum { m_flush       = ((1 << k_flush      ) - 1), s_flush       = (m_flush       << v_flush      ) };
        public:
            /** Construct as ResultDgram(dgram, id). */
            CubeResultDgram(const Pds::EbDgram& dgram, unsigned id) :
                ResultDgram(dgram, id) {}
        public:
            /** Set or clear the record-event bit (auxdata bit 0). */
            void     record       (bool     value) { ADDBITS(record_evt,(value?1:0)); }
            /** Set the bin index (auxdata bits 23-4) to value, masked to 20 bits. */
            void     binIndex     (uint32_t value) { ADDBITS(bin,value); }
            /** Set or clear the record-bin bit (auxdata bit 1). */
            void     updateRecord (bool     value) { ADDBITS(record_bin,(value?1:0)); }
            /** Set or clear the monitor-bin bit (auxdata bit 2). */
            void     updateMonitor(bool     value) { ADDBITS(monitor_bin,(value?1:0)); }
            /** Set or clear the flush bit (auxdata bit 3). */
            void     flush        (bool     value) { ADDBITS(flush,(value?1:0)); }
            /** Return the bin index (auxdata bits 23-4). */
            uint32_t binIndex     () const { return GETBITS(bin); }
            /** Return the record-event bit (auxdata bit 0). */
            bool     record       () const { return GETBITS(record_evt); }
            /** Return the record-bin bit (auxdata bit 1). */
            bool     updateRecord () const { return GETBITS(record_bin); }
            /** Return the monitor-bin bit (auxdata bit 2). */
            bool     updateMonitor() const { return GETBITS(monitor_bin); }
            /** Return the flush bit (auxdata bit 3). */
            bool     flush        () const { return GETBITS(flush); }
            /** Print title followed by the whole object as 32-bit hex words to stdout. */
            void     dump         (const char* title) const {
                printf("%s: CubeResult",title);
                const uint32_t* p = (const uint32_t*)this;
                for(unsigned i=0; i<sizeof(*this)/4; i++)
                    printf(" %08x", p[i]);
                printf("\n");
            }
        };
    };
};

#endif
