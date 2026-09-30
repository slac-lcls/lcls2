#ifndef Pds_Eb_CubeConfigDgram_hh
#define Pds_Eb_CubeConfigDgram_hh

#include "ResultDgram.hh"
#include <stdio.h>

namespace Pds {
    namespace Eb {

        class CubeConfigDgram : public ResultDgram
        {
        public:
            CubeConfigDgram(const Pds::EbDgram& dgram, unsigned id) :
                ResultDgram(dgram, id) {}
        public:
            void     resultType   (char*    rtype);
            void     bins         (unsigned nbins) {
                auxdata(nbins);
            }
            void     appendJson   (char*    json);
            ResultType resultType() const { return (ResultType)monBufNo(); }
            unsigned   bins      () const { return auxdata(); }
            char*      json      () const { return xtc.payload()+sizeof(*this)-sizeof(EbDgram); }
        };
    };
};

#endif
