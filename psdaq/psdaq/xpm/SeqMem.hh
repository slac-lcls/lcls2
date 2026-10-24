/**
 * @file
 * @brief SeqMem, the 2048-word sequence RAM of an XPM sequence engine.
 */
#ifndef Pds_XpmSeqMem_hh
#define Pds_XpmSeqMem_hh

#include "psdaq/cphw/Reg.hh"

namespace Pds {
  namespace Xpm {
    /** Sequence RAM of one engine: 2048 32-bit registers written by XpmSequenceEngine. */
    class SeqMem {
    public:
      /** Return the RAM word index (not range-checked). */
      Cphw::Reg& operator[](unsigned index) { return _word[index]; }
    public:
      Cphw::Reg _word[2048];  ///< The 2048 RAM words.
    };
  };
};

#endif
