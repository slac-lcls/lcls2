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

    class PVSeq
    {
    public:
      PVSeq(XpmSequenceEngine&, const std::string& pvbase);
      ~PVSeq();
    public:
      void cacheSeq         (pvd::shared_vector<const int>&);
      void insertSeq        ();
      void removeSeq        ();
      void scheduleReset    ();
      void forceReset       ();
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
