/**
 * @file
 * @brief TPGen::SequenceEngine, the interface of sequence engines that drive timing patterns for beam requests and experiment control (per the file comment), and InstructionCache.
 */
#ifndef TPG_SequenceEngine_hh
#define TPG_SequenceEngine_hh

//
//  Sequence engine to drive timing patterns for beam requests
//  and experiment control.
//

#include "user_sequence.hh"

#include <vector>

namespace TPGen {
  class Instruction;

  /** Description of one instruction sequence held in sequence RAM, as returned by SequenceEngine::cache(). */
  class InstructionCache {
  public:
    int      index;  ///< Sequence index; -1 if not found.
    unsigned ram_address;  ///< Start address in sequence RAM.
    unsigned ram_size;  ///< Number of RAM words used.
    std::vector<Instruction*> instructions;  ///< The instructions of the sequence (pointers owned by the engine).
  };

  /** Interface of a timing sequence engine: store instruction sequences in sequence RAM and choose where the engine starts or jumps. */
  class SequenceEngine {
  public:
    /** Does nothing; empty virtual destructor. */
    virtual ~SequenceEngine() {}
    //
    //  Insert into the sequence RAM a vector of instructions.
    //  The instructions will be encoded into the hardware representation.
    //  Return value is a (sub)sequence number used to refer to
    //  instruction sequence for the calls that follow.  Many instruction
    //  sequences may be stored in sequence RAM and activated by the setAddress
    //  call below.
    //  Returns negative result on error.
    //
    /** Pure virtual: encode seq into sequence RAM and return its sequence number for the calls below, or a negative value on error (per the interface comment). */
    virtual int  insertSequence(std::vector<Instruction*>& seq)= 0;
    //
    //  Remove a sequence of instructions from sequence RAM clearing
    //  space for more instruction sets.  Should not be performed on a 
    //  sequence currently being executed.
    //  Returns negative result on error.
    //
    /** Pure virtual: remove sequence seq from RAM, freeing its space; must not be used on a running sequence. Returns a negative value on error (per the interface comment). */
    virtual int  removeSequence(int seq)= 0;
    //
    //  Set the starting/jump address for the next reset of the engine.
    //  The address is specified by (sub)sequence index and relative
    //  start offset (instruction index from input vector).
    //
    /** Pure virtual: set the start address for the next reset to instruction start of sequence seq (per the interface comment). */
    virtual void setAddress    (int seq, unsigned start=0, unsigned sync=1)= 0;
    /** Pure virtual: reset the engine. */
    virtual void reset         ()= 0;
    //
    //  Set the starting/jump address for mps faults.
    //  The address is specified by (sub)sequence index and relative
    //  start offset (instruction index from input vector).
    //  The power class is represented by 'mps' and 'pclass'.  The 'mps' index
    //  indicates that this action will be taken when the MPS requests a reduction
    //  to power class='mps'.  The 'pclass' argument indicates the actual power
    //  class of the sequence; 'pclass' must be less than or equal to 'mps'.
    //
    /** Pure virtual: set the jump address used when MPS requests power class mps, as instruction start of sequence seq with power class pclass (per the interface comment). */
    virtual void setMPSJump    (int mps, int seq, unsigned pclass, unsigned start=0)= 0;
    //
    //  Set the starting/jump address for bcs faults.
    //  The address is specified by (sub)sequence index and relative
    //  start offset (instruction index from input vector).
    //  The 'pclass' argument indicates the actual power class of the sequence
    //  (always 0?)
    //
    /** Pure virtual: set the jump address used on a BCS fault, as instruction start of sequence seq with power class pclass (per the interface comment). */
    virtual void setBCSJump    (int seq, unsigned pclass, unsigned start=0)= 0;
    //
    //  Jump to the sequence starting address for the given 'mps' power class.
    //  Note that the MPSJump table (see above) must already be filled for the 'mps' entry.
    //
    /** Pure virtual: jump to the sequence registered for power class mps with setMPSJump() (per the interface comment). */
    virtual void setMPSState   (int mps, unsigned sync=1)= 0;

    //
    //  Return RAM statistics on each instruction set
    //
    /** Pure virtual: return the RAM description of sequence index. */
    virtual InstructionCache              cache(unsigned index) const= 0;
    /** Pure virtual: return the RAM descriptions of all sequences. */
    virtual std::vector<InstructionCache> cache() const= 0;

    /** Pure virtual: print sequence seq. */
    virtual void dumpSequence  (int seq) const= 0;
    /** Pure virtual: print the engine state. */
    virtual void dump          ()        const= 0;
  };
};

#endif
