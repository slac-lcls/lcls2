#ifndef Psdaq_XpmSequenceEngine_hh
#define Psdaq_XpmSequenceEngine_hh

#include "psdaq/xpm/sequence_engine.hh"

namespace Pds {
  namespace Xpm {
    class XpmSequenceEngine : public TPGen::SequenceEngine {
    public:
      void enable        (bool);
    public:
      int  insertSequence(std::vector<TPGen::Instruction*>& seq);
      int  removeSequence(int seq);
      void setAddress    (int seq, unsigned start=0, unsigned sync=1);
      void reset         ();
      void setMPSJump    (int mps, int seq, unsigned pclass, unsigned start=0);
      void setBCSJump    (int seq, unsigned pclass, unsigned start=0);
      void setMPSState   (int mps, unsigned sync=1);
    public:
      void      handle            (unsigned address);
    public:
      TPGen::InstructionCache              cache(unsigned index) const;
      std::vector<TPGen::InstructionCache> cache() const;
      void dumpSequence  (int seq) const;
      void dump          ()        const;
    public:
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

