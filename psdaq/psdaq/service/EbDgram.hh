/**
 * @file
 * @brief PulseId, TimingHeader (the memory-mapped timing header at the start of DMA data) and EbDgram (a Dgram prefixed by the pulse ID word), all packed to 4-byte alignment.
 */
#ifndef PSDAQ_EBDGRAM_H
#define PSDAQ_EBDGRAM_H

#include "xtcdata/xtc/Dgram.hh"

namespace Pds {

#pragma pack(push,4)

class TimingHeader;

/** 64-bit word whose low 56 bits are the pulse ID and whose upper 8 bits carry control bits from the timing system (per the code comment). */
class PulseId {
public:
    /** Store value as the whole 64-bit word, control bits included. */
    PulseId(uint64_t value) : _pulseIdAndControl(value) {}
    // mask off 56 bits, since upper 8 bits can have
    // "control" information from the timing system
    // give methods "timing_" prefix to avoid conflict with
    // methods in Transition
    /** Return the low 56 bits of the word (the pulse ID without the control bits). */
    uint64_t                     pulseId()        const {return _pulseIdAndControl&0x00ffffffffffffff;}
private:
    friend TimingHeader;
    unsigned                     timing_control() const {return (_pulseIdAndControl>>56)&0xff;}
    XtcData::TransitionId::Value timing_service() const {return (XtcData::TransitionId::Value)(timing_control()&0xf);}
protected:
    mutable uint64_t _pulseIdAndControl; // Mutable so the EOL bit can be set
};

/** Timing header at the start of the DMA data: the pulse ID word, the transition fields (time, env), an event counter and two opaque words. Not constructible or copyable, since it is only used over mapped memory (per the code comment). */
class TimingHeader : public PulseId, public XtcData::TransitionBase {
public:
    // Don't allow constructing or copying of the memory mapped class
    /** Deleted: the header is never constructed, only mapped over memory. */
    TimingHeader() = delete;
    /** Deleted: the header cannot be copied. */
    TimingHeader(const TimingHeader&) = delete;
    /** Deleted: the header cannot be copy-assigned. */
    void operator=(const TimingHeader&) = delete;
public:
    /** Return the 8 control bits (bits 63-56 of the pulse ID word). */
    unsigned                      control() const { return timing_control(); }
    /** Return bits 5-4 of the control bits as the transition type. */
    XtcData::TransitionBase::Type type()    const { return XtcData::TransitionBase::Type((control()>>4)&0x3); }
    /** Return the low 4 control bits as the transition ID. */
    XtcData::TransitionId::Value  service() const { return timing_service(); }
    /** Return true if the transition ID is L1Accept. */
    bool                          isEvent() const { return service()==XtcData::TransitionId::L1Accept; }
    /** Return true if bit 7 of the control bits is set (PgpReader logs it as a timing header error). */
    bool                          error()   const { return control() & (1 << 7); }
public:
    uint32_t evtCounter;  ///< Event counter; PgpReader uses its low 24 bits to pick the PGPEvent and to detect jumps.
    uint32_t _opaque[2];  ///< Two 32-bit words not interpreted in this header.
};

/** Dgram prefixed by the pulse ID word, as built by the DRPs and exchanged with the event builders. Bit 62 of the word (control bit 6) marks the end of a batch (EOL). */
class EbDgram : public PulseId, public XtcData::Dgram {
public:
    /** Construct from the pulse ID of pulseId (control bits dropped) and a copy of dgram. */
    EbDgram(const PulseId& pulseId, const Dgram& dgram) : PulseId(pulseId.pulseId()), Dgram(dgram) {}
    /** Construct from a timing header: pulse ID (control bits dropped), time, env set to the control bits in bits 31-24 plus th.env masked by envRogMask, and an empty Parent Xtc with source src. */
    EbDgram(const TimingHeader& th, const XtcData::Src src, uint32_t envRogMask) : PulseId(th.pulseId()) {
        xtc = {{XtcData::TypeId::Parent, 0}, {src}}; // set the src field for the event builders
        time = th.time;
        // move the control bits from the pulseId into the top 8 bits of env.
        env = (th.control()<<24) | (th.env & envRogMask); // filter out other partition ROGs
    }
public:
    /** Set bit 62 of the pulse ID word, marking the last entry of a batch (callable on const objects). */
    void setEOL()  const { _pulseIdAndControl |= 1ULL << (6 + 56); }
    /** Return true if bit 62 of the pulse ID word is set. */
    bool isEOL()   const { return (_pulseIdAndControl & (1ULL << (6 + 56))) != 0; }
    // keepRaw() comes from XtcData::TransitionBase, so TimingHeader has it too
};

#pragma pack(pop)

}

#endif
