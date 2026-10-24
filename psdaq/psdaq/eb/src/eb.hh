/**
 * @file
 * @brief Event-builder constants (limits, timeouts, batch sizes), parameter structures for TEB/MEB contributors and builders, and the latency() helper.
 */
#ifndef Pds_Eb_Eb_hh
#define Pds_Eb_Eb_hh

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <chrono>

#include "xtcdata/xtc/TimeStamp.hh"

#ifndef POSIX_TIME_AT_EPICS_EPOCH
/** Seconds from the POSIX epoch (1970-01-01) to the EPICS epoch (1990-01-01): 631152000. Only defined if not already defined. */
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif


namespace Pds {
  namespace Eb {

    // The following are limited by the number of bits in a uint64_t
    /** Maximum number of contributors (64; the original comment says these limits come from the bits of a uint64_t). */
    const unsigned MAX_DRPS       = 64;         // Max # of Contributors
    /** Maximum number of trigger event builders (4). */
    const unsigned MAX_TEBS       =  4;         // Max # of Event Builders
    /** Maximum number of monitor event builders (4). */
    const unsigned MAX_MEBS       =  4;         // Max # of Monitors
    /** Maximum number of monitor requestors; equal to MAX_MEBS. */
    const unsigned MAX_MRQS       = MAX_MEBS;   // Max # of Monitor Requestors

    /** Number of readout groups supported (8). */
    const unsigned NUM_READOUT_GROUPS = 8;      // # of RoGs supported

    // On picking the following constants:
    // - Determine the contribution arrival time skew that needs to be
    //   accomodated by the TEBs and MEBs.  This determines the event timeout
    //   time EB_TMO_MS.
    // - The TEB event timeout time divided by the pulse ID resolution
    //   (1/TICK_RATE) gives the amount of TEB buffering needed in units of
    //   events.  MAX_LATENCY is this value rounded up to a power of 2.
    // - Batch sizes must also be a power of 2.  In units of events, it is
    //   MAX_ENTRIES.  In units of pulse Id ticks, it is BATCH_DURATION.
    // - When there are multiple TEBs in the system, only one receives an event
    //   batch at a time.  However, if there is a transition in the batch, the
    //   others must also receive it.  The number of transition buffers set
    //   aside for this is determined by the TEB event timeout time multiplied
    //   by the SlowUpdate rate.  This number does not need to be a power of 2.
    //   These buffers are small (sizeof(EbDgram)).
    // - The MEB event timeout time multiplied by the SlowUpdate rate gives the
    //   minimum number of transition buffers needed by the MEB.  This number
    //   does not need to be a power of 2.  MEB transition buffers are large.

    /** System clock rate in Hz according to the original comment; computed in integer arithmetic as 1000000 * 13 / 14. */
    const unsigned TICK_RATE      = 1000000 * 13/14; // System clock rate in Hz

    // Keep *EB_TMO* below control.py's transition and phase 2 timeout
    // Too low results in spurious timeouts and split events
    // - 7 s seems to be too short for UED, so we go back to the previous value:
    /** Event-builder event timeout in milliseconds (12000). A static_assert below requires it to be at most 1000 * MAX_LATENCY / TICK_RATE. */
    const unsigned EB_TMO_MS      = 12000;  // Must be < MAX_LATENCY/TICK_RATE
    /** Number of TEB transition buffers (128); the original comment says it must exceed the timeout times the SlowUpdate rate. */
    const unsigned TEB_TR_BUFFERS = 128;    // # of TEB transition buffers
                                            // > EB_TMO * SlowUpdate rate
    /** Number of MEB transition buffers (24); the original comment says it must exceed the timeout times the SlowUpdate rate. */
    const unsigned MEB_TR_BUFFERS = 24;     // # of MEB transition buffers
                                            // > EB_TMO * SlowUpdate rate

    /** Maximum number of entries (events) per batch (64). */
    const unsigned MAX_ENTRIES    = 64;                        // <= BATCH_DURATION
    /** Batch duration in pulse-ID ticks; equal to MAX_ENTRIES and checked by a static_assert to be a power of 2. */
    const uint64_t BATCH_DURATION = MAX_ENTRIES;               // >= MAX_ENTRIES; power of 2; beam pulse ticks (1 uS)
    //const unsigned MAX_LATENCY    = nextPwrOf2(EB_TMO_MS * TICK_RATE / 1000);
    /** Buffering range in pulse-ID ticks (16 * 1024 * 1024). */
    const unsigned MAX_LATENCY    = 16 * 1024 * 1024;          // In beam pulse ticks (1 uS)
    /** Maximum number of batches in circulation, MAX_LATENCY / MAX_ENTRIES; checked by a static_assert to be a power of 2. */
    const unsigned MAX_BATCHES    = MAX_LATENCY / MAX_ENTRIES; // Max # of batches in circulation

    /** Verbosity levels, from none to most detailed. */
    enum { VL_NONE, /**< Level 0. */ VL_DEFAULT, /**< Level 1. */ VL_BATCH, /**< Level 2. */ VL_EVENT, /**< Level 3. */ VL_DETAILED  /**< Level 4. */ }; // Verbosity levels

    /** Parameters of a TEB contributor (a DRP), per the original comment. */
    struct TebCtrbParams           // Used by TEB contributors (DRPs)
    {
      /** Alias for std::string. */
      using string_t = std::string;
      /** Alias for `std::vector<std::string>`. */
      using vecstr_t = std::vector<std::string>;
      /** Alias for a std::string to std::string map. */
      using kwmap_t  = std::map<std::string,std::string>;

      /** Network interface to use. */
      string_t ifAddr;             // Network interface to use
      /** Port served to receive results. */
      string_t port;               // Served port to receive results
      /** Instrument name, used for monitoring. */
      string_t instrument;         // Instrument name for monitoring
      /** Partition (platform) number of the chosen system. */
      unsigned partition;          // The chosen system
      /** Unique name passed on the command line. */
      string_t alias;              // Unique name passed on cmd line
      /** Detector name. */
      string_t detName;            // The detector name
      /** Detector segment number. */
      unsigned detSegment;         // The detector segment
      /** Contributor instance identifier. */
      unsigned id;                 // Contributor instance identifier
      /** Bit list of the IDs of the event builders. */
      uint64_t builders;           // ID bit list of EBs
      /** TEB addresses. */
      vecstr_t addrs;              // TEB addresses
      /** TEB ports. */
      vecstr_t ports;              // TEB ports
      /** Maximum size of a contribution. */
      size_t   maxInputSize;       // Max size of contribution
      /** Level of detail to print (mutable, so it can be changed through a const reference). */
      mutable
      unsigned verbose;            // Level of detail to print
      /** Readout group that receives trigger result data. */
      uint16_t readoutGroup;       // RO group receiving trigger result data
      /** Readout group that supplies trigger input data. */
      uint16_t contractor;         // RO group supplying trigger input  data
      /** Maximum entries per batch; the original comment says to set it to 1 to disable batching. */
      unsigned maxEntries;         // Set to 1 to disable batching
      /** Keyword arguments. */
      kwmap_t  kwargs;             // Keyword arguments
    };

    /** Parameters of an MEB contributor (a DRP), per the original comment. */
    struct MebCtrbParams           // Used by MEB contributors (DRPs)
    {
      /** Alias for std::string. */
      using string_t  = std::string;
      /** Alias for `std::vector<std::string>`. */
      using vecstr_t  = std::vector<std::string>;
      /** Alias for `std::vector<unsigned>`. */
      using vecuint_t = std::vector<unsigned>;
      /** Alias for a std::string to std::string map. */
      using kwmap_t   = std::map<std::string,std::string>;

      /** MEB addresses. */
      vecstr_t  addrs;             // MEB addresses
      /** MEB ports. */
      vecstr_t  ports;             // MEB ports
      /** Instrument name, used for monitoring. */
      string_t  instrument;        // Instrument name for monitoring
      /** Partition (platform) number of the chosen system. */
      unsigned  partition;         // The chosen system
      /** Unique name passed on the command line. */
      string_t  alias;             // Unique name passed on cmd line
      /** Detector name. */
      string_t  detName;           // The detector name
      /** Detector segment number. */
      unsigned  detSegment;        // The detector segment
      /** Contributor instance identifier. */
      unsigned  id;                // Contributor instance identifier
      /** Maximum number of events to provide buffers for (one entry per MEB). */
      vecuint_t maxEvents;         // Max # of events to provide for
      /** Maximum event size. */
      size_t    maxEvSize;         // Max event size
      /** Maximum non-event (transition) size. */
      size_t    maxTrSize;         // Max non-event size
      /** Level of detail to print (mutable, so it can be changed through a const reference). */
      mutable
      unsigned  verbose;           // Level of detail to print
      /** Keyword arguments. */
      kwmap_t   kwargs;            // Keyword arguments
    };

    /** Parameters used by both TEBs and MEBs, per the original comment. */
    struct EbParams                // Used with both TEBs and MEBs
    {
      /** Alias for std::string. */
      using string_t  = std::string;
      /** Alias for `std::vector<std::string>`. */
      using vecstr_t  = std::vector<std::string>;
      /** Alias for `std::vector<size_t>`. */
      using vecsize_t = std::vector<size_t>;
      /** Alias for `std::vector<unsigned>`. */
      using vecuint_t = std::vector<unsigned>;
      /** Array of NUM_READOUT_GROUPS uint64_t values (one per readout group). */
      using u64arr_t  = std::array<uint64_t, NUM_READOUT_GROUPS>;
      /** Alias for a std::string to std::string map. */
      using kwmap_t   = std::map<std::string,std::string>;

      /** Network interface to use. */
      string_t  ifAddr;            // Network interface to use
      /** Event-builder port to serve. */
      string_t  ebPort;            // EB port to serve
      /** Port on which monitor requests are received. */
      string_t  mrqPort;           // Mon request port to receive on
      /** Instrument name, used for monitoring. */
      string_t  instrument;        // Instrument name for monitoring
      /** Collection server. */
      string_t  collSrv;           // Collection server
      /** Partition (platform) number of the chosen system. */
      unsigned  partition;         // The chosen system
      /** Unique name passed on the command line. */
      string_t  alias;             // Unique name passed on cmd line
      /** Event-builder instance identifier. */
      unsigned  id;                // EB instance identifier
      /** Bit list of all readout groups in use. */
      unsigned  rogs;              // Bit list of all readout groups in use
      /** Bit list of contributor IDs. */
      uint64_t  contributors;      // ID bit list of contributors
      /** Contributors that provide the buffer index for results. */
      uint64_t  indexSources;      // Sources providing buffer index for Results
      /** Per readout group, the contributors that provide inputs. */
      u64arr_t  contractors;       // Ctrbs providing Inputs  per readout group
      /** Per readout group, the contributors that expect results. */
      u64arr_t  receivers;         // Ctrbs expecting Results per readout group
      /** Contributor addresses. */
      vecstr_t  addrs;             // Contributor addresses
      /** Contributor ports. */
      vecstr_t  ports;             // Contributor ports
      /** Maximum non-event EbDgram size for each contributor. */
      vecsize_t maxTrSize;         // Max non-event EbDgram size for each Ctrb
      /** Maximum number of entries per batch. */
      unsigned  maxEntries;        // Max number of entries per batch
      /** Limit of the results buffer index range. */
      unsigned  maxBuffers;        // Limit of Results buffer index range
      /** Number of input buffers for each DRP. */
      vecuint_t numBuffers;        // Number of Inputs buffers for each DRP
      /** Number of monitor request servers. */
      unsigned  numMrqs;           // Number of Mon request servers
      /** Number of monitor event buffers for each MEB. */
      vecuint_t numMebEvBufs;      // Number of Mon event buffers for each MEB
      /** Prometheus configuration location for run-time monitoring. */
      string_t  prometheusDir;     // Run-time monitoring prometheus config file
      /** Unique DRP names taken from the cnf id field. */
      vecstr_t  drps;              // Unique DRP names from cnf id field
      /** Keyword arguments. */
      kwmap_t   kwargs;            // Keyword arguments
      /** Level of detail to print (mutable, so it can be changed through a const reference). */
      mutable
      unsigned  verbose;           // Level of detail to print
    };

    /**
     * Return the time from timestamp time to now (std::chrono::system_clock) as a count of duration type T.
     * The timestamp is moved to the POSIX epoch by adding a function-static offset that starts at POSIX_TIME_AT_EPICS_EPOCH; if the difference still exceeds that many seconds, the offset is increased by the difference and the computation is repeated.
     */
    template<typename T>
    int64_t latency(const XtcData::TimeStamp&);

    // Sanity checks
    static_assert((BATCH_DURATION & (BATCH_DURATION - 1)) == 0, "BATCH_DURATION must be a power of 2");
    static_assert((MAX_BATCHES & (MAX_BATCHES - 1)) == 0, "MAX_BATCHES must be a power of 2");
    static_assert((EB_TMO_MS <= 1000ull * MAX_LATENCY/TICK_RATE), "EB_TMO_MS is too large");
  };
};

template<typename T>
int64_t Pds::Eb::latency(const XtcData::TimeStamp& time)
{
  using std::chrono::system_clock;
  using std::chrono::duration_cast;
  using sec_t = std::chrono::seconds;
  using ns_t  = std::chrono::nanoseconds;
  static unsigned long _tOffset = POSIX_TIME_AT_EPICS_EPOCH; // 20 years

  // XPMs on external timing produce times since the 1990 epoch
  // XPMs on internal timing reset time to 0 when they're started
  auto now = system_clock::now();               // 1970 epoch (Takes a long time!)
  auto dgt = sec_t{ time.seconds() + _tOffset } // Convert to 1970 epoch
           + ns_t { time.nanoseconds() };
  system_clock::time_point tp{ duration_cast<system_clock::duration>(dgt) };
  auto dt =  now - tp;

  // If dt is still large, increase tOffset to get differences near zero
  if (duration_cast<sec_t>(dt).count() > POSIX_TIME_AT_EPICS_EPOCH)
  {
    _tOffset += duration_cast<sec_t>(dt).count();
    return latency<T>(time);
  }
  return duration_cast<T>(dt).count();
}

#endif
