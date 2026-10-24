/**
 * @file
 * @brief SeqState, the state registers of one XPM sequence engine.
 */
//
//  Builder for the SeqState application registers
//
#ifndef SeqState_hh
#define SeqState_hh

#include "psdaq/cphw/Reg.hh"

namespace Pds {
  namespace Xpm {
    /** State registers of one sequence engine, printed by XpmSequenceEngine::dump(). */
    class SeqState {
    public:
      Cphw::Reg countRequests;  ///< Register printed as Req by XpmSequenceEngine::dump().
      Cphw::Reg countInvalid;  ///< Register printed as Inv by XpmSequenceEngine::dump().
      Cphw::Reg address;  ///< Register printed as Addr by XpmSequenceEngine::dump().
      Cphw::Reg condcnt;  ///< Register printed as Cond by XpmSequenceEngine::dump().
      /** Return the value of condcnt converted to a pointer (the register value itself, not the address of the register). */
      const uint8_t* condCount() const { 
        uint32_t v = condcnt;
        return reinterpret_cast<const uint8_t*>(v);
      }
    };
  };
};

#endif
