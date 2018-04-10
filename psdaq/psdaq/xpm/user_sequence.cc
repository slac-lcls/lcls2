#include "user_sequence.hh"

using namespace TPGen;

Instruction::Type ControlRequest::instr() const
{ return Instruction::Request; }

BeamRequest::BeamRequest(unsigned q) : charge(q) {}
BeamRequest::~BeamRequest() {}
ControlRequest::Type BeamRequest::request() const 
{ return ControlRequest::Beam; }
unsigned BeamRequest::value() const
{ return charge; }

ExptRequest::ExptRequest(unsigned w) : word(w) {}
ExptRequest::~ExptRequest() {}
ControlRequest::Type ExptRequest::request() const 
{ return ControlRequest::Expt; }
unsigned ExptRequest::value() const
{ return word; }

Checkpoint::Checkpoint(Callback* cb) : _callback(cb) {}
Checkpoint::~Checkpoint() { delete _callback; }
Callback* Checkpoint::callback() const { return _callback; }
Instruction::Type Checkpoint::instr() const
{ return Instruction::Check; }

FixedRateSync::FixedRateSync(unsigned        m,
			     unsigned        o) :
  marker_id  (m),
  occurrence(o)
{}

FixedRateSync::~FixedRateSync()
{
}

Instruction::Type FixedRateSync::instr() const
{ return Instruction::Fixed; }


ACRateSync::ACRateSync(unsigned        t,
		       unsigned        m,
		       unsigned        o) :
  timeslot_mask(t),
  marker_id    (m),
  occurrence   (o)
{
}

ACRateSync::~ACRateSync() 
{
}

Instruction::Type ACRateSync::instr() const
{ return Instruction::AC; }


Branch::Branch( unsigned  a ) :
  address(a),
  counter(ctrA),
  test   (0)
{
}

Branch::Branch( unsigned  a,     // address to jump to if test fails
		CCnt      c,     // index of counter to test and increment
		unsigned  t ) :  // value to test against
  address(a),
  counter(c),
  test   (t)
{
}

Branch::~Branch() {}

Instruction::Type Branch::instr() const
{ return Instruction::Branch; }
