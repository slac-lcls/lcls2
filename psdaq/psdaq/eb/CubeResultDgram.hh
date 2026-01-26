#ifndef Pds_Eb_CubeResultDgram_hh
#define Pds_Eb_CubeResultDgram_hh

#include "ResultDgram.hh"

#define ADDBITS(FLD,VAL) auxdata((auxdata()&~s_##FLD) | ((VAL&m_##FLD)<<v_##FLD))
#define GETBITS(FLD)     (auxdata()&~s_##FLD)>>v_##FLD

namespace Pds {
    namespace Eb {

        class CubeResultDgram : public ResultDgram
        {
            enum { v_bin    =  0, k_bin    = 10 };
            enum { v_worker = 10, k_worker =  6 };
            enum { v_record = 16, k_record =  1 };

            enum { m_bin    = ((1 < k_bin   ) - 1), s_bin    = (m_bin    << v_bin   ) };
            enum { m_worker = ((1 < k_worker) - 1), s_worker = (m_worker << v_worker) };
            enum { m_record = ((1 < k_record) - 1), s_record = (m_record << v_record) };
        public:
            CubeResultDgram(const Pds::EbDgram& dgram, unsigned id) :
                ResultDgram(dgram, id)
            {}
        public:
            //void     bin   (uint32_t value) { ADDBITS(bin   ,value); }
            void     bin   (uint32_t value) { auxdata( (auxdata()&~s_bin) | ((value<<v_bin)&s_bin)); }
            void     worker(uint32_t value) { ADDBITS(worker,value); }
            void     record(bool     value) { ADDBITS(record,(value?1:0)); }
            //uint32_t bin   () const { return GETBITS(bin); }
            uint32_t bin   () const { return (auxdata()&s_bin)>>v_bin; }
            uint32_t worker() const { return GETBITS(worker); }
            bool     record() const { return GETBITS(record); }
        };
    };
};

#undef ADDBITS
#undef GETBITS

#endif
