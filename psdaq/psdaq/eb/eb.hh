#ifndef Pds_Eb_Eb_hh
#define Pds_Eb_Eb_hh

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <map>


namespace Pds {
  namespace Eb {

    // The following are limited by the number of bits in a uint64_t
    const unsigned MAX_DRPS       = 64;         // Max # of Contributors
    const unsigned MAX_TEBS       =  4;         // Max # of Event Builders
    const unsigned MAX_MEBS       =  4;         // Max # of Monitors
    const unsigned MAX_MRQS       = MAX_MEBS;   // Max # of Monitor Requestors

    const unsigned NUM_READOUT_GROUPS = 8;      // # of RoGs supported

    // On picking the following constants:
    // - Determine the contribution arrival time skew that needs to be
    //   accomodated by the TEBs and MEBs.  This determines the event timeout
    //   times TEB_TMO_MS and MEB_TMO_MS.
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

    const unsigned TICK_RATE      = 928500; // System clock rate in Hz

    // Keep *EB_TMO* below control.py's transition and phase 2 timeout
    // Too low results in spurious timeouts and split events
    // - 7 s seems to be too short for UED, so we go back to the previous value:
    const unsigned TEB_TMO_MS     = 12000;  // Must be < MAX_LATENCY/TICK_RATE
    const unsigned MEB_TMO_MS     = 12000;  // <= TEB_TMO_MS
    const unsigned TEB_TR_BUFFERS = 128;    // # of TEB transition buffers
                                            // > TEB_TMO * SlowUpdate rate
    const unsigned MEB_TR_BUFFERS = 24;     // # of MEB transition buffers
                                            // > MEB_TMO * SlowUpdate rate

    const unsigned MAX_ENTRIES    = 64;                        // <= BATCH_DURATION
    const uint64_t BATCH_DURATION = MAX_ENTRIES;               // >= MAX_ENTRIES; power of 2; beam pulse ticks (1 uS)
    //const unsigned MAX_LATENCY    = nextPwrOf2(TEB_TMO_MS * TICK_RATE / 1000);
    const unsigned MAX_LATENCY    = 16 * 1024 * 1024;          // In beam pulse ticks (1 uS)
    const unsigned MAX_BATCHES    = MAX_LATENCY / MAX_ENTRIES; // Max # of batches in circulation

    enum { VL_NONE, VL_DEFAULT, VL_BATCH, VL_EVENT, VL_DETAILED }; // Verbosity levels

    struct TebCtrbParams           // Used by TEB contributors (DRPs)
    {
      using string_t = std::string;
      using vecstr_t = std::vector<std::string>;
      using kwmap_t  = std::map<std::string,std::string>;

      string_t ifAddr;             // Network interface to use
      string_t port;               // Served port to receive results
      string_t instrument;         // Instrument name for monitoring
      unsigned partition;          // The chosen system
      string_t alias;              // Unique name passed on cmd line
      string_t detName;            // The detector name
      unsigned detSegment;         // The detector segment
      unsigned id;                 // Contributor instance identifier
      uint64_t builders;           // ID bit list of EBs
      vecstr_t addrs;              // TEB addresses
      vecstr_t ports;              // TEB ports
      size_t   maxInputSize;       // Max size of contribution
      int      core[2];            // Cores to pin threads to
      mutable
      unsigned verbose;            // Level of detail to print
      uint16_t readoutGroup;       // RO group receiving trigger result data
      uint16_t contractor;         // RO group supplying trigger input  data
      unsigned maxEntries;         // Set to 1 to disable batching
      kwmap_t  kwargs;             // Keyword arguments
    };

    struct MebCtrbParams           // Used by MEB contributors (DRPs)
    {
      using vecstr_t = std::vector<std::string>;
      using string_t = std::string;
      using kwmap_t  = std::map<std::string,std::string>;

      vecstr_t addrs;              // MEB addresses
      vecstr_t ports;              // MEB ports
      string_t instrument;         // Instrument name for monitoring
      unsigned partition;          // The chosen system
      string_t alias;              // Unique name passed on cmd line
      string_t detName;            // The detector name
      unsigned detSegment;         // The detector segment
      unsigned id;                 // Contributor instance identifier
      unsigned maxEvents;          // Max # of events to provide for
      size_t   maxEvSize;          // Max event size
      size_t   maxTrSize;          // Max non-event size
      mutable
      unsigned verbose;            // Level of detail to print
      kwmap_t  kwargs;             // Keyword arguments
    };

    struct EbParams                // Used with both TEBs and MEBs
    {
      using string_t  = std::string;
      using vecstr_t  = std::vector<std::string>;
      using vecsize_t = std::vector<size_t>;
      using vecuint_t = std::vector<unsigned>;
      using u64arr_t  = std::array<uint64_t, NUM_READOUT_GROUPS>;
      using kwmap_t   = std::map<std::string,std::string>;

      string_t  ifAddr;            // Network interface to use
      string_t  ebPort;            // EB port to serve
      string_t  mrqPort;           // Mon request port to receive on
      string_t  instrument;        // Instrument name for monitoring
      unsigned  partition;         // The chosen system
      string_t  alias;             // Unique name passed on cmd line
      unsigned  id;                // EB instance identifier
      unsigned  rogs;              // Bit list of all readout groups in use
      uint64_t  contributors;      // ID bit list of contributors
      uint64_t  indexSources;      // Sources providing buffer index for Results
      u64arr_t  contractors;       // Ctrbs providing Inputs  per readout group
      u64arr_t  receivers;         // Ctrbs expecting Results per readout group
      vecstr_t  addrs;             // Contributor addresses
      vecstr_t  ports;             // Contributor ports
      vecsize_t maxTrSize;         // Max non-event EbDgram size for each Ctrb
      unsigned  maxEntries;        // Max number of entries per batch
      unsigned  maxBuffers;        // Limit of Results buffer index range
      vecuint_t numBuffers;        // Number of Inputs buffers for each DRP
      unsigned  numMrqs;           // Number of Mon request servers
      unsigned  numMebEvBufs;      // Number of Mon event buffers
      string_t  prometheusDir;     // Run-time monitoring prometheus config file
      kwmap_t   kwargs;            // Keyword arguments
      int       core[2];           // Cores to pin threads to
      mutable
      unsigned  verbose;           // Level of detail to print
    };

    // Sanity checks
    static_assert((BATCH_DURATION & (BATCH_DURATION - 1)) == 0, "BATCH_DURATION must be a power of 2");
    static_assert((MAX_BATCHES & (MAX_BATCHES - 1)) == 0, "MAX_BATCHES must be a power of 2");
    static_assert((TEB_TMO_MS <= 1000ull * MAX_LATENCY/TICK_RATE), "TEB_TMO_MS is too large");
  };
};

#endif
