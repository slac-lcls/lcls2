#ifndef TPG_UserSequence_hh
#define TPG_UserSequence_hh

#include <stdint.h>

namespace TPGen {
  class Callback;

  //
  //  Generic instruction for sequence engine
  //
  class Instruction {
  public:
    enum Type { Fixed, AC, Branch, Check, Request };
    virtual ~Instruction() {}
  public:
    virtual Instruction::Type instr() const = 0;
  };

  //
  //  Checkpoint instruction to notify software 
  //
  class Checkpoint : public Instruction {
  public:
    Checkpoint(Callback*);
    ~Checkpoint();
  public:
    Instruction::Type instr() const;
    Callback* callback() const;
  private:
    Callback* _callback;
  };

  //
  //  Sync to n-th occurrence of fixed rate marker 
  //  (and insert request)
  //
  class FixedRateSync : public Instruction {
  public:
    FixedRateSync(unsigned        marker_id,
		  unsigned        occurrence);
    ~FixedRateSync();
  public:
    Instruction::Type instr() const;
  public:
    unsigned        marker_id;
    unsigned        occurrence;
  };

  //  Sync to n-th occurrence of powerline-synchronized marker 
  //  (and insert request)
  //
  class ACRateSync : public Instruction {
  public:
    ACRateSync(unsigned        timeslot_mask,
	       unsigned        marker_id,
	       unsigned        occurrence);
    ~ACRateSync();
  public:
    Instruction::Type instr() const;
  public:
    unsigned        timeslot_mask;
    unsigned        marker_id;
    unsigned        occurrence;
  };

  enum CCnt {ctrA, ctrB, ctrC, ctrD}; // conditional counter

  class Branch : public Instruction {
  public:
    //
    //  Unconditional jump
    //
    Branch( unsigned  address );   // address to jump to unconditionally
    //
    //  Jump when counter is less than test
    //
    Branch( unsigned  address,     // address to jump to if test fails
	    CCnt      counter,     // index of counter to test and increment
	    unsigned  test );      // value to test against
	    
    ~Branch();
  public:
    Instruction::Type instr() const;
  public:
    unsigned  address;
    CCnt      counter;
    unsigned  test;
  };

  class ControlRequest : public Instruction {
  public:
    enum Type { Beam, Expt };
    virtual ~ControlRequest() {}
  public:
    Instruction::Type instr() const;
    virtual ControlRequest::Type request() const = 0;
    virtual unsigned value() const = 0;
  };

  //
  //  Request beam to destination with selected charge
  //
  class BeamRequest : public ControlRequest {
  public:
    BeamRequest(unsigned charge);
    ~BeamRequest();
  public:
    ControlRequest::Type request() const;
    unsigned value() const;
  public:
    unsigned charge;
  };

  //
  //  Request experiment control word
  //
  class ExptRequest : public ControlRequest {
  public:
    ExptRequest(uint32_t);
    ~ExptRequest();
  public:
    ControlRequest::Type request() const;
    unsigned value() const;
  public:
    uint32_t word;
  };
};

#endif
