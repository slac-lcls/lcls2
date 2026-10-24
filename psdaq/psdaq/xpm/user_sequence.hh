/**
 * @file
 * @brief Instruction classes of the timing sequence engine (TPGen): fixed-rate and AC-rate syncs, branches, checkpoints and control requests.
 */
#ifndef TPG_UserSequence_hh
#define TPG_UserSequence_hh

//#define EXCLUDE_CHECKPOINT

#include <stdint.h>

/** Namespace of the timing pattern sequence engine interface and its instruction classes. */
namespace TPGen {
  //
  //  Generic instruction for sequence engine
  //
  /** Base of the sequence engine instructions (per the code comment); instr() identifies the concrete type. */
  class Instruction {
  public:
    /** Instruction type returned by instr(). */
    enum Type { Fixed, /**< FixedRateSync. */ AC, /**< ACRateSync. */ Branch, /**< Branch. */ Check, /**< Checkpoint. */ Request  /**< ControlRequest (BeamRequest or ExptRequest). */ };
    /** Does nothing; empty virtual destructor. */
    virtual ~Instruction() {}
  public:
    /** Pure virtual: return the instruction type. */
    virtual Instruction::Type instr() const = 0;
  };

#ifndef EXCLUDE_CHECKPOINT
  class Callback;

  //
  //  Checkpoint instruction to notify software 
  //
  /** Instruction to notify software through a callback (per the code comment); compiled unless EXCLUDE_CHECKPOINT is defined. */
  class Checkpoint : public Instruction {
  public:
    /** Keep the callback pointer (not owned). */
    Checkpoint(Callback*);
    /** Does nothing; the callback is not deleted. */
    ~Checkpoint();
  public:
    /** Return Instruction::Check. */
    Instruction::Type instr() const;
    /** Return the callback pointer given to the constructor. */
    Callback* callback() const;
  private:
    Callback* _callback;
  };
#endif

  //
  //  Sync to n-th occurrence of fixed rate marker 
  //  (and insert request)
  //
  /** Wait for the occurrence-th fixed-rate marker marker_id (per the code comment). */
  class FixedRateSync : public Instruction {
  public:
    /** Store marker_id and occurrence. */
    FixedRateSync(unsigned        marker_id,
		  unsigned        occurrence);
    /** Does nothing; empty body. */
    ~FixedRateSync();
  public:
    /** Return Instruction::Fixed. */
    Instruction::Type instr() const;
  public:
    unsigned        marker_id;  ///< Fixed-rate marker number; XpmSequenceEngine encodes 4 bits of it.
    unsigned        occurrence;  ///< Number of markers to wait for; XpmSequenceEngine encodes 12 bits of it.
  };

  //  Sync to n-th occurrence of powerline-synchronized marker 
  //  (and insert request)
  //
  /** Wait for the occurrence-th powerline-synchronized marker marker_id in the timeslots of timeslot_mask (per the code comment). */
  class ACRateSync : public Instruction {
  public:
    /** Store timeslot_mask, marker_id and occurrence. */
    ACRateSync(unsigned        timeslot_mask,
	       unsigned        marker_id,
	       unsigned        occurrence);
    /** Does nothing; empty body. */
    ~ACRateSync();
  public:
    /** Return Instruction::AC. */
    Instruction::Type instr() const;
  public:
    unsigned        timeslot_mask;  ///< Timeslot mask; XpmSequenceEngine encodes 6 bits of it.
    unsigned        marker_id;  ///< AC-rate marker number; XpmSequenceEngine encodes 4 bits of it.
    unsigned        occurrence;  ///< Number of markers to wait for; XpmSequenceEngine encodes 12 bits of it.
  };

  /** Conditional counter used by Branch (per the code comment). */
  enum CCnt {ctrA, /**< Counter A (0); also the counter of an unconditional Branch. */ ctrB, /**< Counter B (1). */ ctrC, /**< Counter C (2). */ ctrD /**< Counter D (3). */ }; // conditional counter

  /** Jump to another instruction, unconditionally or, per the constructor comment, while a counter is less than a test value. */
  class Branch : public Instruction {
  public:
    //
    //  Unconditional jump
    //
    /** Unconditional jump to address (counter ctrA, test 0). */
    Branch( unsigned  address );   // address to jump to unconditionally
    //
    //  Jump when counter is less than test
    //
    /** Conditional jump: per the code comments, jump to address while counter (which is tested and incremented) is less than test. */
    Branch( unsigned  address,     // address to jump to if test fails
	    CCnt      counter,     // index of counter to test and increment
	    unsigned  test );      // value to test against
	    
    /** Does nothing; empty body. */
    ~Branch();
  public:
    /** Return Instruction::Branch. */
    Instruction::Type instr() const;
  public:
    unsigned  address;  ///< Target address; XpmSequenceEngine treats it as an instruction index within the sequence.
    CCnt      counter;  ///< Counter tested by a conditional branch.
    unsigned  test;  ///< Test value; 0 means an unconditional branch to XpmSequenceEngine.
  };

  /** Base of the request instructions; value() is the request word. */
  class ControlRequest : public Instruction {
  public:
    /** Request type returned by request(). */
    enum Type { Beam, /**< BeamRequest. */ Expt  /**< ExptRequest. */ };
    /** Does nothing; empty virtual destructor. */
    virtual ~ControlRequest() {}
  public:
    /** Return Instruction::Request. */
    Instruction::Type instr() const;
    /** Pure virtual: return the request type. */
    virtual ControlRequest::Type request() const = 0;
    /** Pure virtual: return the request word. */
    virtual unsigned value() const = 0;
  };

  //
  //  Request beam to destination with selected charge
  //
  /** Request beam with the selected charge (per the code comment). XpmSequenceEngine::insertSequence() rejects it. */
  class BeamRequest : public ControlRequest {
  public:
    /** Store charge. */
    BeamRequest(unsigned charge);
    /** Does nothing; empty body. */
    ~BeamRequest();
  public:
    /** Return ControlRequest::Beam. */
    ControlRequest::Type request() const;
    /** Return charge. */
    unsigned value() const;
  public:
    unsigned charge;  ///< Requested charge value.
  };

  //
  //  Request experiment control word
  //
  /** Request an experiment control word (per the code comment). */
  class ExptRequest : public ControlRequest {
  public:
    /** Store the word. */
    ExptRequest(uint32_t);
    /** Does nothing; empty body. */
    ~ExptRequest();
  public:
    /** Return ControlRequest::Expt. */
    ControlRequest::Type request() const;
    /** Return word. */
    unsigned value() const;
  public:
    uint32_t word;  ///< Experiment control word; XpmSequenceEngine writes it into a request instruction.
  };
};

#endif
