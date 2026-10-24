/**
 * @file
 * @brief XpmSequenceEngine, the SequenceEngine implementation for one XPM sequence engine (sequence RAM, jump and state registers).
 */
#ifndef Psdaq_XpmSequenceEngine_hh
#define Psdaq_XpmSequenceEngine_hh

#include "psdaq/xpm/sequence_engine.hh"

namespace Pds {
  namespace Xpm {
    /** SequenceEngine for one XPM sequence engine: encodes instructions into its sequence RAM and controls its enable, restart and start-address registers. Created through Module::sequenceEngine(). */
    class XpmSequenceEngine : public TPGen::SequenceEngine {
    public:
      /** Set or clear the bit of this engine in the sequence enable register. */
      void enable        (bool);
    public:
      /** Encode seq into the smallest free RAM block that fits, remember it (the engine takes ownership of the instructions) and return its index (0 to 63). Returns -1 for a non-Expt request or no RAM space, -2 if all indices are used, -3 for a branch beyond the sequence. */
      int  insertSequence(std::vector<TPGen::Instruction*>& seq);
      /** Free sequence seq: delete its instructions, write 0 to its first RAM word and forget it. Returns 0, or -1 if seq is not in use, -2 if it is not found. */
      int  removeSequence(int seq);
      /** If sequence seq exists, set the manual start address to its instruction start and the manual sync value to sync. */
      void setAddress    (int seq, unsigned start=0, unsigned sync=1);
      /** Write the bit of this engine to the restart register. */
      void reset         ();
      /** Does nothing; empty body. */
      void setMPSJump    (int mps, int seq, unsigned pclass, unsigned start=0);
      /** Does nothing; empty body. */
      void setBCSJump    (int seq, unsigned pclass, unsigned start=0);
      /** Does nothing; empty body. */
      void setMPSState   (int mps, unsigned sync=1);
    public:
      /** Does nothing; empty body. */
      void      handle            (unsigned address);
    public:
      /** Return the RAM description of sequence index, or one with index -1 if it does not exist. */
      TPGen::InstructionCache              cache(unsigned index) const;
      /** Return the RAM descriptions of all stored sequences, including the two trap sequences made by the constructor. */
      std::vector<TPGen::InstructionCache> cache() const;
      /** Print the RAM words of sequence seq; does nothing if seq is not in use. */
      void dumpSequence  (int seq) const;
      /** Print the request, invalid, address and condition registers of this engine and the RAM words of every sequence in use. */
      void dump          ()        const;
    public:
      /** Set the file-static verbosity of the engine code. */
      static void verbosity(unsigned);
    protected:
      //
      //  Construct an engine with its sequence RAM and start register address
      //
      XpmSequenceEngine(void*,
                        unsigned);
      ~XpmSequenceEngine();
    protected:
      friend class Module;
      class PrivateData;
      PrivateData* _private;
    };
  };
};

#endif

