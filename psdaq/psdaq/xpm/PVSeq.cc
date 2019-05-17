#include "psdaq/xpm/PVSeq.hh"
#include "psdaq/xpm/XpmSequenceEngine.hh"
#include "psdaq/epicstools/PVBase.hh"

using Pds_Epics::PVBase;
using Pds::Xpm::PVSeq;

//  Mirrors sequser.py
enum Instructions { FixedRateSync, 
                    ACRateSync,
                    BranchInstr, 
                    CheckPointInstr, 
                    BeamRequestInstr, 
                    ControlRequestInstr };

namespace Pds {
  namespace Xpm {

#define PV(name) name ## PV

#define CPV(name, updatedBody, connectedBody)                           \
                                                                        \
    class PV(name) : public PVBase                                      \
    {                                                                   \
    public:                                                             \
      PV(name)(PVSeq& parent, const char* pvName, unsigned idx = 0) :   \
        PVBase(pvName),                                                 \
        _parent(parent),                                                \
        _idx(idx) {}                                                    \
      virtual ~PV(name)() {}                                            \
    public:                                                             \
      void updated  () { updatedBody }                                  \
      void onConnect() { connectedBody }                                \
    private:                                                            \
      PVSeq& _parent;                                                   \
      unsigned _idx;                                                    \
    };

    CPV(INSTRS,
        { pvd::shared_vector<const int> instrs;
          getVectorAs<int>(instrs);
          _parent.cacheSeq(instrs); },  // should update INSTRCNT
        { putFrom<unsigned>(0); })
    CPV(INS,
        { if (getScalarAs<unsigned>()!=0) _parent.insertSeq(); },  // should update SEQ00IDX/DESC
        { putFrom<unsigned>(0); })
    CPV(RMVSEQ,
        { if (getScalarAs<unsigned>()!=0) _parent.removeSeq(); },
        { putFrom<unsigned>(0); })
    CPV(SCHEDRESET,
        { if (getScalarAs<unsigned>()!=0) _parent.scheduleReset(); },
        { putFrom<unsigned>(0); })
    CPV(FORCERESET,
        { if (getScalarAs<unsigned>()!=0) _parent.forceReset(); },
        { putFrom<unsigned>(0); })

    class SeqHandle {
    public:
      SeqHandle(unsigned i, 
                unsigned a,
                const std::string& d) : index(i), addr(a), desc(d) {}
      unsigned    index;
      unsigned    addr;
      std::string desc;
    };
  };
};

enum { DESCINSTRS = 0,
       INSTRCNT,
       SEQIDX,
       SEQDESC,
       SEQ00IDX,
       SEQ00DESC,
       RMVIDX,
       RUNIDX, };

PVSeq::PVSeq(XpmSequenceEngine& eng, 
             const std::string& pvbase) :
  _eng(eng), _pv(0), _desc(64), _seq(0)
{
  //  Don't need to monitor these
#define NPV(name)  _pv.push_back( new PVBase((pvbase+":"+#name).c_str()) )
  NPV(DESCINSTRS);
  NPV(INSTRCNT);
  NPV(SEQIDX);
  NPV(SEQDESC);
  NPV(SEQ00IDX);
  NPV(SEQ00DESC);
  NPV(RMVIDX);
  NPV(RUNIDX);
  //  Must monitor these
#undef NPV
#define NPV(name)  _pv.push_back( new PV(name)(*this, (pvbase+":"+#name).c_str()) )
  NPV(INSTRS);
  NPV(RMVSEQ);
  NPV(INS);
  NPV(SCHEDRESET);
  NPV(FORCERESET);

  _desc[0] = "Reserved";
  _desc[1] = "Reserved";
  for(unsigned i=2; i<64; i++)
    _desc[i] = "";

  _pv[SEQ00IDX]->putFrom<int>(0);
}
  
PVSeq::~PVSeq() 
{
  for(unsigned i=0; i<_pv.size(); i++)
    delete _pv[i];
  _pv.resize(0);

  for(unsigned i=0; i<_seq.size(); i++)
    delete _seq[i];
  _seq.resize(0);
}

void PVSeq::cacheSeq(pvd::shared_vector<const int>& instrs)
{
  // Parse the PV data into a list of instructions
  std::vector<TPGen::Instruction*>& seq = _seq;
  for(unsigned i=0; i<seq.size(); i++)
    delete seq[i];
  seq.resize(0);

  unsigned i=0;
  unsigned ninstr = instrs[i++];
  while( i<instrs.size() && seq.size()<ninstr ) {
    unsigned nargs = instrs[i++];
    switch( instrs[i] ) {
    case FixedRateSync:
      seq.push_back(new TPGen::FixedRateSync(instrs[i+1],instrs[i+2]));
      break;
    case ACRateSync:
      seq.push_back(new TPGen::ACRateSync(instrs[i+1],instrs[i+2],instrs[i+3]));
      break;
    case BranchInstr:
      seq.push_back(nargs==1 ? 
                    new TPGen::Branch(instrs[i+1]) :
                    new TPGen::Branch(instrs[i+1],(TPGen::CCnt)instrs[i+2],instrs[i+3]) );
      break;
      //    case TPGen::Instruction::Check:
    case ControlRequestInstr:
      seq.push_back(new TPGen::ExptRequest(instrs[i+1]));
      break;
    default:
      printf("Unknown instruction 0x%x (%u)\n",instrs[i],i);
      break;
    }
    i += 6;
  }

  printf("Setting INSTRCNT %zu/%u\n",seq.size(),ninstr);
  _pv[INSTRCNT]->putFrom<unsigned>(seq.size());
}

void PVSeq::insertSeq()
{
  std::vector<TPGen::Instruction*>& seq = _seq;
  int rval = _eng.insertSequence(seq);  // transfer ownership of instructions
  if (rval < 0) {
    printf("Insert sequence failed [%d]\n", rval);
    return;
  }
  seq.resize(0);

  _pv[SEQ00IDX]->putFrom<int>(rval);

  _eng.dump  ();

  //  Can't seem to use getScalar<std::string> or putFrom<std::string>
  // std::string d = _pv[DESCINSTRS]->getScalarAsString();
  // printf("Retrieved DESCINSTRS [%s]\n", d.c_str());
  // _pv[SEQ00DESC]->putFrom<std::string>( d );
}

void PVSeq::removeSeq()
{
  if (!_pv[RMVIDX]->connected()) {
    printf("Remove index PV not connected\n");
    return;
  }

  unsigned i = _pv[RMVIDX]->getScalarAs<unsigned>();
  if (i > 1 && i < _desc.size()) {
    _eng.removeSequence(i);
    _desc[i] = "";
    _pv[SEQ00IDX]->putFrom<int>(0);
    //  StringArray support?
    // pvd::shared_vector<std::string> v;
    // for(unsigned j=0; j<64; j++)
    //   v[j] = _desc[j];
    // _pv[SEQDESC]->putFromVector(v);
  }
  else 
    printf("Ignored attempt to remove sub-sequence index %d\n", i);
}

void PVSeq::scheduleReset()
{
  if (!_pv[RUNIDX]->connected()) {
    printf("Run index PV not connected\n");
    return;
  }

  unsigned i = _pv[RUNIDX]->getScalarAs<unsigned>();
  printf("Scheduling index %u\n",i);
  _eng.enable(true);
  _eng.setAddress(i,0,1);
  _eng.reset ();
}

void PVSeq::forceReset()
{
  if (!_pv[RUNIDX]->connected()) {
    printf("Run index PV not connected\n");
    return;
  }

  unsigned i = _pv[RUNIDX]->getScalarAs<unsigned>();
  printf("Starting index %u\n",i);
  _eng.enable(true);
  _eng.setAddress(i,0,0);
  _eng.reset ();
}

