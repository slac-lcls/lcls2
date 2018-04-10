#include "psdaq/xpm/XpmSequenceEngine.hh"
#include "psdaq/xpm/SeqState.hh"
#include "psdaq/xpm/SeqJump.hh"
#include "psdaq/xpm/SeqMem.hh"
#include "psdaq/cphw/Reg.hh"

#include <climits>
#include <map>

using TPGen::InstructionCache;
using TPGen::Instruction;
using TPGen::Checkpoint;
using TPGen::FixedRateSync;
using TPGen::ACRateSync;
using TPGen::Branch;
using TPGen::BeamRequest;
using TPGen::ControlRequest;
using TPGen::ExptRequest;

namespace Pds {
  namespace Xpm {
    class SeqRegs {
    public:
      uint16_t SeqAddrLen () const { return unsigned(resources)&0xfff; }
      uint8_t  NControlSeq() const { return (unsigned(resources)>>16)&0xff; }
      uint8_t  NXpmSeq    () const { return (unsigned(resources)>>24)&0xff; }
      Cphw::Reg resources;
      Cphw::Reg SeqEnable;
      Cphw::Reg SeqRestart;
      uint32_t reserved_C[(0x80-0xC)>>2];

      SeqState seqState[16];
      uint32_t reserved_4000[(0x3f80-16*sizeof(SeqState))>>2];
      
      SeqJump  seqJump[16];
      uint32_t reserved_8000[(0x4000-16*sizeof(SeqJump))>>2];

      SeqMem   seqMem[2];
    };

    class SeqCache {
    public:
      int      index;
      unsigned size;
      std::vector<Instruction*> instr;
    };

    class XpmSequenceEngine::PrivateData {
    public:
      PrivateData(void*    p,
                  unsigned id) :
        _regs   ( reinterpret_cast<SeqRegs*>(p) ),
        _jump   ( _regs->seqJump[id] ),
        _ram    ( _regs->seqMem [id] ),
        _id     (id),
        _indices(0) {}
    public:
      SeqRegs*                     _regs;
      SeqJump&                     _jump;
      SeqMem&                      _ram;
      uint32_t                     _id;
      uint64_t                     _indices;  // bit mask of reserved indices
      std::map<unsigned,SeqCache>  _caches;   // map start address to sequence
    };
  };
};

using namespace Pds::Xpm;

static unsigned _verbose=0;

/**
   --
   --  Sequencer instructions:
   --    (31:29)="010"  Fixed Rate Sync -- shifted down 1
   --       (19:16)=marker_id
   --       (11:0)=occurrence
   --    (31:29)="011"  AC Rate Sync -- shifted down 1
   --       (28:23)=timeslot_mask  -- shifted down 1
   --       (19:16)=marker_id
   --       (11:0)=occurrence
   --    (31:29)="001"  Checkpoint/Notify -- shifted down 1
   --    (31:29)="000"  Branch -- shifted down 1
   --       (28:27)=counter
   --       (24)=conditional
   --       (23:16)=test_value
   --       (10:0)=address
   --    (31:29)="100" Request
   --       (15:0)  Value
**/

static unsigned _nwords(const Instruction& i) { return 1; }

static inline uint32_t _word(const FixedRateSync& i)
{
  return (2<<29) | ((i.marker_id&0xf)<<16) | (i.occurrence&0xfff);
}

static inline uint32_t _word(const ACRateSync& i)
{
  return (3<<29) | ((i.timeslot_mask&0x3f)<<23) | ((i.marker_id&0xf)<<16) | (i.occurrence&0xfff);
}

static inline uint32_t _word(const Branch& i, unsigned a)
{
  return (i.test&0xff)==0 ? (a&0x3ff) :
    ((unsigned(i.counter)&0x3)<<27) | (1<<24) | ((i.test&0xff)<<16) | (a&0x3ff);
}

static inline uint32_t _word(const Checkpoint& i)
{
  return 1<<29;
}

static inline uint32_t _word(const ControlRequest& i)
{
  return (4<<29) | i.value();
}

/**
static uint32_t _request(const ControlRequest* req, unsigned id)
{
  uint32_t v = 0;
  if (req) {
    switch(req->request()) {
    case ControlRequest::Beam:
      v = static_cast<const BeamRequest*>(req)->charge&0xffff;
      break;
    case ControlRequest::Expt:
      v = static_cast<const ExptRequest*>(req)->word;
      break;
    default:
      break;
    }
  }
  return v; 
}
**/

static int _lookup_address(const std::map<unsigned,SeqCache>& caches,
			   int seq, 
			   unsigned start)
{
  for(std::map<unsigned,SeqCache>::const_iterator it=caches.begin();
      it!=caches.end(); it++)
    if (it->second.index == seq) {
      unsigned a = it->first;
      for(unsigned i=0; i<start; i++)
	a += _nwords(*it->second.instr[i]);
      return a;
    }
  return -1;
}

XpmSequenceEngine::XpmSequenceEngine(void* p, unsigned id) :
  _private( new XpmSequenceEngine::PrivateData(p, id) )
{
  _private->_indices = 3;

  //  Assign a single instruction sequence at first and last address to trap
  std::vector<Instruction*> v(1);
  v[0] = new Branch(0);
  unsigned a=0;
  _private->_caches[a].index = 0;
  _private->_caches[a].size  = 1;
  _private->_caches[a].instr = v;
  _private->_ram   [a] = _word(*static_cast<const Branch*>(v[0]),a);

  unsigned addrWidth = _private->_regs->SeqAddrLen();
  v[0] = new Branch(0);
  a=(1<<addrWidth)-1;
  _private->_caches[a].index = 1;
  _private->_caches[a].size  = 1;
  _private->_caches[a].instr = v;
  _private->_ram   [a] = _word(*static_cast<const Branch*>(v[0]),a);
}

XpmSequenceEngine::~XpmSequenceEngine()
{
  for(std::map<unsigned,SeqCache>::iterator it=_private->_caches.begin();
      it!=_private->_caches.end(); it++)
    removeSequence(it->second.index);
  delete _private;
}

int  XpmSequenceEngine::insertSequence(std::vector<Instruction*>& seq)
{
  int rval=0, aindex=-3;

  do {
    //  Validate sequence
    for(unsigned i=0; i<seq.size(); i++) {
      if (seq[i]->instr()==Instruction::Request) {
        const ControlRequest* request = static_cast<const ControlRequest*>(seq[i]);
        if (request->request()!=ControlRequest::Expt) {
          printf("Invalid request: instr %x, type %x, value %x\n", 
                 (unsigned)request->instr(), 
                 (unsigned)request->request(), 
                 request->value());
          rval=-1;
        }
      }
    }

    if (rval) break;

    //  Calculate memory needed
    unsigned nwords=0;
    for(unsigned i=0; i<seq.size(); i++)
      nwords += _nwords(*seq[i]);

    if (_verbose>1)
      printf("insertSequence %zu instructions, %u words\n",seq.size(),nwords);

    //  Find memory range (just) large enough
    unsigned best_ram=0;
    {
      unsigned addr=0;
      unsigned best_size=INT_MAX;
      for(std::map<unsigned,SeqCache>::iterator it=_private->_caches.begin();
	  it!=_private->_caches.end(); it++) {
	unsigned isize = it->first-addr;
	if (_verbose>1)
	  printf("Found memblock %x:%x [%x]\n",addr,it->first,isize);
	if (isize == nwords) {
	  best_size = isize;
	  best_ram = addr;
	  break;
	}
	else if (isize>nwords && isize<best_size) {
	  best_size = isize;
	  best_ram = addr;
	}
	addr = it->first+it->second.size;
      }
      if (best_size==INT_MAX) {
        printf("BRAM space unavailable\n");
	rval=-1;  // no space available in BRAM
	break;
      }
      if (_verbose>1)
	printf("Using memblock %x:%x [%x]\n",best_ram,best_ram+nwords,nwords);
    }

    if (rval) break;

    //  Cache instruction vector, start address (reserve memory)
    if (_private->_indices == -1ULL) {
      rval=-2;
      break;
    }

    for(unsigned i=0; i<64; i++)
      if ((_private->_indices & (1<<i))==0) {
	_private->_indices |= (1<<i);
	aindex=i;
	break;
      }
  
    _private->_caches[best_ram].index = aindex;
    _private->_caches[best_ram].size  = nwords;
    _private->_caches[best_ram].instr = seq;
  
    //  Translate addresses
    unsigned addr = best_ram;
    for(unsigned i=0; i<seq.size(); i++) {
      switch(seq[i]->instr()) {
      case Instruction::Branch:
	{ const Branch& instr = *static_cast<const Branch*>(seq[i]);
	  int jumpto = instr.address;
	  if (jumpto > int(seq.size())) rval=-3;
	  else if (jumpto >= 0) {
	    unsigned jaddr = 0;
	    for(int j=0; j<jumpto; j++)
	      jaddr += _nwords(*seq[j]);
	    _private->_ram[addr++] = _word(instr,jaddr+best_ram);
	  }
	} break;
      case Instruction::Fixed:
	_private->_ram[addr++] =
	  _word(*static_cast<const FixedRateSync*>(seq[i])); 
	break;
      case Instruction::AC:
	_private->_ram[addr++] = _word(*static_cast<const ACRateSync*>(seq[i]));
	break;
      case Instruction::Check:
	{ const Checkpoint& instr = 
	    *static_cast<const Checkpoint*>(seq[i]);
	  _private->_ram[addr++] = _word(instr); }
	break;
      case Instruction::Request:
	_private->_ram[addr++] =
	  _word(*static_cast<const ControlRequest*>(seq[i]));
        break;
      default:
	break;
      }
    }
    if (rval)
      removeSequence(aindex);
  } while(0);

  return rval ? rval : aindex;
}

int  XpmSequenceEngine::removeSequence(int index)
{
  if ((_private->_indices&(1ULL<<index))==0) return -1;
  _private->_indices &= ~(1ULL<<index);

  //  Lookup sequence
  for(std::map<unsigned,SeqCache>::iterator it=_private->_caches.begin();
      it!=_private->_caches.end(); it++)
    if (it->second.index == index) {
      //  Free instruction vector
      for(unsigned i=0; i<it->second.instr.size(); i++)
	delete it->second.instr[i];

      //  Trap entry and free memory
      _private->_ram[it->first] = 0;
      _private->_caches.erase(it);

      return 0;
    }
  return -2;
}

void XpmSequenceEngine::setAddress    (int seq, unsigned start, unsigned sync)
{
  int a = _lookup_address(_private->_caches,seq,start);
  if (a>=0) {
    _private->_jump.setManStart(a,0);
    _private->_jump.setManSync (sync);
  }
}

void XpmSequenceEngine::enable        (bool e)
{
  uint32_t v = _private->_regs->SeqEnable;
  if (e)
    v |= (1<<_private->_id);
  else
    v &= ~(1<<_private->_id);
  _private->_regs->SeqEnable = v;
}

void XpmSequenceEngine::reset         ()
{
  uint32_t v = (1<<_private->_id);
  _private->_regs->SeqRestart = v;
}

void XpmSequenceEngine::setMPSJump    (int mps, int seq, unsigned pclass, unsigned start)
{
}

void XpmSequenceEngine::setBCSJump    (int seq, unsigned pclass, unsigned start)
{
}

void XpmSequenceEngine::setMPSState  (int mps, unsigned sync)
{
}


void XpmSequenceEngine::dumpSequence(int index) const
{
  if ((_private->_indices&(1ULL<<index))==0) return;

  //  Lookup sequence
  for(std::map<unsigned,SeqCache>::const_iterator it=_private->_caches.begin();
      it!=_private->_caches.end(); it++)
    if (it->second.index == index)
      for(unsigned i=0; i<it->second.size; i++)
        printf("[%08x] %08x\n",it->first+i,unsigned(_private->_ram[it->first+i]));
}

void XpmSequenceEngine::handle(unsigned addr)
{
}

void XpmSequenceEngine::dump() const
{
  for(unsigned i=0; i<64; i++)
    if ((_private->_indices&(1ULL<<i))!=0) {
      printf("Sequence %d\n",i);
      dumpSequence(i);
    }
}

void XpmSequenceEngine::verbosity(unsigned v)
{ _verbose=v; }

InstructionCache XpmSequenceEngine::cache(unsigned index) const
{
  for(std::map<unsigned,SeqCache>::iterator it=_private->_caches.begin();
      it!=_private->_caches.end(); it++) {
    if (it->second.index == (int)index) {
      InstructionCache c;
      c.index        = it->second.index;
      c.ram_address  = it->first;
      c.ram_size     = it->second.size;
      c.instructions = it->second.instr;
      return c;
      break;
    }
  }
  //  throw std::invalid_argument("index not found");
  InstructionCache c;
  c.index        = -1;
  return c;
}

std::vector<InstructionCache> XpmSequenceEngine::cache() const
{
  std::vector<InstructionCache> rval;
  for(std::map<unsigned,SeqCache>::iterator it=_private->_caches.begin();
      it!=_private->_caches.end(); it++) {
    InstructionCache c;
    c.index        = it->second.index;
    c.ram_address  = it->first;
    c.ram_size     = it->second.size;
    c.instructions = it->second.instr;
    rval.push_back(c);
  }
  return rval;
}
