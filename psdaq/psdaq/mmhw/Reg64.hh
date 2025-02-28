#ifndef Mmhw_Reg64_hh
#define Mmhw_Reg64_hh

#include "psdaq/mmhw/Reg.hh"
#include <stdint.h>

//
//  Maps memory accesses into dmaRead(Write)Register calls
//
namespace Pds {
    namespace Mmhw {
        class Reg64 {
        public:
            Reg64& operator=(const unsigned long);
            operator unsigned long() const;
        public:
            void setBit  (unsigned);
            void clearBit(unsigned);
        private:
            Reg _lower;
            Reg _upper;
        };
    };
};

#endif
