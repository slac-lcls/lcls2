/**
 * @file
 * @brief PVSeq, the EPICS PVs that load, remove and start instruction sequences on an XPM sequence engine.
 */
#ifndef Xpm_PVSeq_hh
#define Xpm_PVSeq_hh

#include "psdaq/epicstools/EpicsPVA.hh"

#include <string>
#include <vector>

namespace TPGen { class Instruction; };

namespace Pds {

  namespace Xpm {
    
    class XpmSequenceEngine;
    class SeqHandle;

    /** EPICS PVs of one XPM sequence engine, named pvbase, a colon and the PV name. Writing a non-zero value to INS, RMVSEQ, SCHEDRESET or FORCERESET inserts, removes or starts a sequence; INSTRS receives the encoded instructions. Created by PVCtrls::allocate() with pvbase ending in SEQENG:0. */
    class PVSeq
    {
    public:
      /** Create the PVs for pvbase (DESCINSTRS, INSTRCNT, SEQIDX, SEQDESC, SEQ00IDX, SEQ00DESC, RMVIDX, RUNIDX and RUNNING, plus the monitored INSTRS, RMVSEQ, INS, SCHEDRESET and FORCERESET). Mark descriptions 0 and 1 as reserved and write 0 to SEQ00IDX. */
      PVSeq(XpmSequenceEngine&, const std::string& pvbase);
      /** Delete the PVs and any parsed instructions not yet inserted. */
      ~PVSeq();
    public:
      /** Replace the parsed instructions with those decoded from the INSTRS array and write their number to INSTRCNT. Element 0 is the instruction count; each instruction then takes 7 ints (argument count, opcode as in psdaq/seq/seq.py, arguments). Unknown opcodes, including beam requests, are skipped with a message. */
      void cacheSeq         (pvd::shared_vector<const int>&);
      /** Insert the parsed instructions into the engine and, on success, write the returned index to SEQ00IDX, give up the instructions (the engine owns them) and dump the engine. On failure, print the error code and keep the instructions. */
      void insertSeq        ();
      /** Remove the sequence whose index is in RMVIDX if that index is 2 to 63, clear its description and write 0 to SEQ00IDX. Prints a message and does nothing if RMVIDX is not connected or the index is out of range. */
      void removeSeq        ();
      /** Start the sequence whose index is in RUNIDX: write RUNNING (1 if the index is above 1, else 0), enable the engine, set its start address with sync value 1 and reset it. Does nothing if RUNIDX is not connected. */
      void scheduleReset    ();
      /** Same as scheduleReset() but sets the start address with sync value 0. */
      void forceReset       ();
      /** Write 0 to RUNNING; the argument is not used. */
      void checkPoint       (unsigned);
    private:
      XpmSequenceEngine&                _eng;
      std::vector<Pds_Epics::EpicsPVA*> _pv;
      std::vector<std::string>          _desc;
      std::vector<TPGen::Instruction*>  _seq;
    };
  };
};

#endif
