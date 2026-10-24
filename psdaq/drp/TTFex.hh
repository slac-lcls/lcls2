/**
 * @file
 * @brief Helper types for timetool feature extraction: Roi, EventInfo (decoded timing information), EventSelect (event-code and destination filter) and the COPY_CFG_VECTOR macro.
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "xtcdata/xtc/Array.hh"

namespace XtcData { class DescData; }

namespace Drp {

    /** Rectangular region of interest with inclusive corners; OpalTTFex.cc reads it from the configuration entries name.x0, name.y0, name.x1 and name.y1 and uses it for projections. */
    class Roi {
    public:
        /** Inclusive bounds: columns x0 to x1 and rows y0 to y1 (OpalTTFex.cc truncates x1 and y1 to the frame size). */
        unsigned x0, y0, /**< Top row of the region (inclusive). */ x1, /**< Right column of the region (inclusive; truncated to the frame width by OpalTTFex.cc). */ y1;  ///< Bottom row of the region (inclusive; truncated to the frame height by OpalTTFex.cc).
    };

    //
    //  The (l2si-core) EventTimingMessage auxiliary data
    //
    /** Timing information of one event, decoded from the timing data (the code comment calls it the l2si-core EventTimingMessage auxiliary data): pulse ID, fixed-rate and AC-rate bits, beam present and destination, and the sequencer words. */
    class EventInfo {
    public:
        /** Construct without initializing any field. */
        EventInfo() {}
        /** Fill the fields from the timing data in descdata: pulseId, the fixedRates (10) and acRates (6) flag arrays packed into bit masks, ebeamPresent, the low 4 bits of ebeamDestn and the 18 sequenceValues words. _timeSlots is not set. */
        EventInfo(XtcData::DescData&);
    public:
        /** Return true if bit rate of the fixed-rate mask is set. */
        bool fixedRate(unsigned rate) const { return _fixedRates & (1<<rate); }
        /** Return true if bit rate of the AC-rate mask is set and the time-slot mask shares a bit with tslots. */
        bool acRate(unsigned rate,
                    unsigned tslots) const { return _acRates & (1<<rate) && (_timeSlots&tslots); }
        /** Return true if bit (ec & 0xf) of sequencer word ec >> 4 is set; ec 0 returns false (the code comment calls it do-not-care). */
        bool eventCode(unsigned ec) const { 
            if (ec==0) return 0; // special case to indicate we don't care
            return _seqInfo[ec>>4] & (1<<(ec&0xf)); }
        /** Return true if bit bit of sequencer word word is set. */
        bool sequencer(unsigned word,
                       unsigned bit) const { return _seqInfo[word] & (1<<bit); }
        /** Return true if the beam is present and the beam destination equals dest; dest 0 returns false (do-not-care). */
        bool destination(unsigned dest) const { 
            if (dest==0) return 0; // special case to indicate we don't care
            return _beamPresent && (_beamDestn == dest); }
    public:
        uint64_t _pulseId;  ///< Pulse ID, from the pulseId value.
        unsigned _fixedRates  : 10;  ///< Fixed-rate flags as a 10-bit mask (bit i set when fixedRates[i] is non-zero).
        unsigned _acRates     : 6;  ///< AC-rate flags as a 6-bit mask (bit i set when acRates[i] is non-zero).
        unsigned _timeSlots   : 8;  ///< Time-slot mask (8 bits) tested by acRate(); the DescData constructor does not set it.
        /** 1 if ebeamPresent was non-zero (one bit; the 3 unnamed bits after it are padding). */
        unsigned _beamPresent : 1, : 3;  ///< 1 if ebeamPresent was non-zero; followed by 3 unnamed padding bits.
        unsigned _beamDestn   : 4;  ///< Beam destination (low 4 bits of ebeamDestn).
        uint16_t _seqInfo[18];  ///< The 18 16-bit sequencer words (sequenceValues) used by eventCode() and sequencer().
    };

    /** Event filter on event codes and beam destinations: an event is selected if it matches an include condition and no exclude condition; a value of 0 is never matched. */
    class EventSelect {
    public:
        /** Set the include and exclude event codes and destinations to 0 (unused). */
        EventSelect() : incl_eventcode(0), incl_destination(0),
                        excl_eventcode(0), excl_destination(0) {}
    public:
        /** Return true if info has the include event code or the include destination, and has neither the exclude event code nor the exclude destination. */
        bool select(const EventInfo&) const;
    public:
        /** Set the event code that selects an event (0 never matches). */
        void set_incl_eventcode  (unsigned code) { incl_eventcode = code; }
        /** Set the event code that rejects an event (0 never matches). */
        void set_excl_eventcode  (unsigned code) { excl_eventcode = code; }
        /** Set the beam destination that selects an event (0 never matches). */
        void set_incl_destination(unsigned dest) { incl_destination = dest; } 
        /** Set the beam destination that rejects an event (0 never matches). */
        void set_excl_destination(unsigned dest) { excl_destination = dest; } 
    private:
        unsigned incl_eventcode;
        unsigned incl_destination;
        unsigned excl_eventcode;
        unsigned excl_destination;
    };
};

//
//  Some helpful macros when iterating over a configuration
//
/** For use inside a loop over configuration names with name, descdata and i in scope: if the current name equals a, copy array entry i of element type t into the vector b (resized to fit). */
#define COPY_CFG_VECTOR(t,a,b)                                  \
    if (strcmp(name.name(),#a)==0) {                            \
        Array<t> w = descdata.get_array<t>(i);                  \
        std::vector<t>& v = b;                                  \
        unsigned len = w.num_elem();                            \
        v.resize(len);                                          \
        memcpy(v.data(),w.data(),len*sizeof(t));                \
    }
        
