/**
 * @file
 * @brief CubeConfigDgram, a ResultDgram that carries a result type, a bin count and appended JSON text.
 */
#ifndef Pds_Eb_CubeConfigDgram_hh
#define Pds_Eb_CubeConfigDgram_hh

#include "ResultDgram.hh"
#include <stdio.h>

namespace Pds {
    namespace Eb {

        /** ResultDgram that stores a ResultType in the monitor buffer number word, a bin count in the auxdata field, and NUL-terminated JSON text appended to its payload. */
        class CubeConfigDgram : public ResultDgram
        {
        public:
            /** Construct as ResultDgram(dgram, id). */
            CubeConfigDgram(const Pds::EbDgram& dgram, unsigned id) :
                ResultDgram(dgram, id) {}
        public:
            /** Store Cube if rtype is the string Cube, Window if it is Window, otherwise Base, in the monitor buffer number word. */
            void     resultType   (char*    rtype);
            /** Store nbins in the auxdata field (24 bits). */
            void     bins         (unsigned nbins) {
                auxdata(nbins);
            }
            /** Append the string json, including its terminating NUL, to the Xtc payload (Xtc::alloc() without an end check). */
            void     appendJson   (char*    json);
            /** Return the monitor buffer number word cast to ResultType. */
            ResultType resultType() const { return (ResultType)monBufNo(); }
            /** Return the auxdata field (the bin count set by bins(unsigned)). */
            unsigned   bins      () const { return auxdata(); }
            /** Return a pointer to the payload bytes just after the ResultDgram fields, which is where the first appendJson() call places its text. */
            char*      json      () const { return xtc.payload()+sizeof(*this)-sizeof(EbDgram); }
        };
    };
};

#endif
