/**
 * @file
 * @brief Declares the datagram header classes TransitionBase, Transition, Dgram and L1Dgram (packed with 4-byte alignment).
 */
#ifndef XtcData_Dgram_hh
#define XtcData_Dgram_hh

#include "TimeStamp.hh"
#include "TransitionId.hh"
#include "Xtc.hh"
#include <stdint.h>

#pragma pack(push,4)

namespace XtcData
{

/**
 * Datagram header base: a TimeStamp followed by a 32-bit env word.
 * The constructor packs the Type into env bits 31:28, the TransitionId into bits 27:24 and the low 24 bits of env_ into bits 23:0.
 */
class TransitionBase {
public:
    /** Transition type; Transition::type() reads it from env bits 29:28. */
    enum Type { Event = 0, /**< Value 0. */ Occurrence = 1, /**< Value 1. */ Marker = 2  /**< Value 2. */ };
    /** Count of Type values. */
    enum { NumberOfTypes = 3  /**< Value 3, the number of Type values. */ };
    /** Default constructor; time and env are left uninitialized. */
    TransitionBase() {}
    /** Set time to time_ and env to (type_<<28)|(tid_<<24)|(env_&0xffffff). */
    TransitionBase(Type type_, TransitionId::Value tid_,
                   const TimeStamp& time_, uint32_t env_) :
        time(time_), env((type_<<28)|(tid_<<24)|(env_&0xffffff)) {}
public:
    /** Return the low 16 bits of env. */
    uint16_t readoutGroups() const { return (env)&0xffff; }
    /// L0Raw, b6 of the reserved field L1Dgram::reserved() describes: the timing
    /// system marks an event for raw data retention, at a rate set per readout group.
    bool     keepRaw()       const { return (env>>22)&1; }
public:
    TimeStamp time;  ///< Datagram timestamp.
    uint32_t env;  ///< Packed word: type, transition id and 24 low bits (layout in the class description).
};

/** TransitionBase with accessors that decode fields of the env word. */
class Transition : public TransitionBase {
public:
    /** Default constructor; fields are left uninitialized as in TransitionBase(). */
    Transition() {}
    /** Forward all arguments to TransitionBase(type_, tid_, time_, env_). */
    Transition(Type type_, TransitionId::Value tid_,
               const TimeStamp& time_, uint32_t env_) :
      TransitionBase(type_, tid_, time_, env_) {}
public:
    /** Return env bits 31:24. */
    unsigned control()            const { return (env>>24)&0xff; }
    /** Return env bits 27:24 as a TransitionId::Value. */
    TransitionId::Value service() const { return TransitionId::Value(control()&0xf); }
    /** Return env bits 29:28 as a Type. */
    Type type()                   const { return Type((control()>>4)&0x3); }
    /** Return true if service() is TransitionId::L1Accept. */
    bool isEvent()                const { return service()==TransitionId::L1Accept; }
    /** Return true if env bit 16 is set and isEvent() is false. */
    bool isExtended()             const { return ((env>>16)&1) && !isEvent(); }
};

/** Datagram: the Transition header immediately followed by the top-level Xtc, whose payload follows it in memory. */
class Dgram : public Transition {
public:
    static const unsigned MaxSize = 0x1000000;  ///< 0x1000000. Inferred from the name to be a maximum datagram size in bytes; not enforced in this header.
    /** Default constructor; header fields are uninitialized and xtc is default-constructed (damage 0, extent 0). */
    Dgram() {}
    /** Copy the header from transition_; xtc is default-constructed (damage 0, extent 0). */
    Dgram(const Transition& transition_) :
        Transition(transition_) { }
    /** Copy the header from transition_ and copy-construct xtc from xtc_. The Xtc copy constructor copies src, damage and contains but sets extent to sizeof(Xtc). */
    Dgram(const Transition& transition_, const Xtc& xtc_) :
        Transition(transition_), xtc(xtc_)  { }
public:
    Xtc xtc;  ///< Top-level Xtc container.
};

/** Dgram with an accessor for env bits 23:16. */
class L1Dgram : public Dgram {
public:
    // 8 reserved bits.  Perhaps for future trigger lines?
    // b0   - L0Accept
    // b5:1 - L0Tag
    // b6   - L0Raw      -- TransitionBase::keepRaw()
    // b7   - L0Reject
    /** Return env bits 23:16. The source comment labels them b0 L0Accept, b5:1 L0Tag, b6 L0Raw, b7 L0Reject. */
    uint16_t reserved() const { return (env>>16)&0xff; }
};

}

#pragma pack(pop)

#endif
