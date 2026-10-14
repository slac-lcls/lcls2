#ifndef Pds_Eb_CubeConfigDgram_hh
#define Pds_Eb_CubeConfigDgram_hh

#include "CubeResultDgram.hh"

namespace Pds {
    namespace Eb {

        class CubeConfigDgram : public CubeResultDgram
        {
        public:
            CubeConfigDgram(const Pds::EbDgram& dgram, unsigned id) :
                CubeResultDgram(dgram, id) {}
        public:
            void     bins         (unsigned nbins) {
                binIndex(nbins-1);
            }
            void     appendJson   (char*    json) {
                unsigned len = strlen(json)+1;
                memcpy( xtc.alloc(len, 0), json, len);
            }
        };
    };
};

#endif
