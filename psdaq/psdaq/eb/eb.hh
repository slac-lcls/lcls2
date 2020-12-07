#ifndef Pds_Eb_Eb_hh
#define Pds_Eb_Eb_hh

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <map>


namespace Pds {
  namespace Eb {

    const unsigned MAX_DRPS       = 64;         // Maximum possible number of Contributors
    const unsigned MAX_TEBS       =  4;         // Maximum possible number of Event Builders
    const unsigned MAX_MEBS       =  4;         // Maximum possible number of Monitors
    const unsigned MAX_MRQS       = MAX_MEBS;   // Maximum possible number of Monitor Requestors

    const unsigned TICK_RATE      = 928500;     // System clock rate in Hz

    const unsigned MAX_ENTRIES    = 64;                        // <= BATCH_DURATION
    const uint64_t BATCH_DURATION = MAX_ENTRIES;               // >= MAX_ENTRIES; power of 2; beam pulse ticks (1 uS)
    const unsigned MAX_LATENCY    = 16 * 1024 * 1024;          // In beam pulse ticks (1 uS)
    const unsigned MAX_BATCHES    = MAX_LATENCY / MAX_ENTRIES; // Max # of batches in circulation

    const unsigned EB_TMO_MS      = 1000ul * MAX_LATENCY/TICK_RATE - 2000; // ms

    const unsigned NUM_READOUT_GROUPS     = 16; // # of RoGs supported
    const unsigned NUM_TRANSITION_BUFFERS =  8; // # of buffers for implementing
                                                // flow control for transitions
                                                // (not batches) transferred between
                                                // DRPs and EBs; May be different
                                                // from DRP's para.nTrBuffers

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
      bool     batching;           // Batching enable flag
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
      using u64arr_t  = std::array<uint64_t, NUM_READOUT_GROUPS>;
      using kwmap_t   = std::map<std::string,std::string>;

      string_t  ifAddr;            // Network interface to use
      string_t  ebPort;            // EB port to serve
      string_t  mrqPort;           // Mon request port to receive on
      string_t  instrument;        // Instrument name for monitoring
      unsigned  partition;         // The chosen system
      string_t  alias;             // Unique name passed on cmd line
      unsigned  id;                // EB instance identifier
      uint64_t  contributors;      // ID bit list of contributors
      u64arr_t  contractors;       // Ctrbs providing Inputs  per readout group
      u64arr_t  receivers;         // Ctrbs expecting Results per readout group
      vecstr_t  addrs;             // Contributor addresses
      vecstr_t  ports;             // Contributor ports
      vecsize_t maxTrSize;         // Max non-event EbDgram size for each Ctrb
      size_t    maxResultSize;     // Max result EbDgram size
      unsigned  numMrqs;           // Number of Mon request servers
      string_t  prometheusDir;     // Run-time monitoring prometheus config file
      kwmap_t   kwargs;            // Keyword arguments
      int       core[2];           // Cores to pin threads to
      mutable
      unsigned  verbose;           // Level of detail to print
    };

    // Sanity checks
    static_assert((BATCH_DURATION & (BATCH_DURATION - 1)) == 0, "BATCH_DURATION must be a power of 2");
    static_assert((MAX_BATCHES & (MAX_BATCHES - 1)) == 0, "MAX_BATCHES must be a power of 2");
  };
};

#endif
