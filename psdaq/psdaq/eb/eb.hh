#ifndef Pds_Eb_Eb_hh
#define Pds_Eb_Eb_hh

#include <cstdint>
#include <string>
#include <vector>
#include <array>


namespace Pds {
  namespace Eb {

    const unsigned RTMON_PORT_BASE = 5559;

    const unsigned MAX_DRPS       = 64;         // Maximum possible number of Contributors
    const unsigned MAX_TEBS       = 64;         // Maximum possible number of Event Builders
    const unsigned MAX_MEBS       = 64;         // Maximum possible number of Monitors

    const unsigned TEB_PORT_BASE  = 32768;                     // TEB to receive L3 contributions
    const unsigned DRP_PORT_BASE  = TEB_PORT_BASE + MAX_TEBS;  // TEB to send    results
    const unsigned MRQ_PORT_BASE  = DRP_PORT_BASE + MAX_DRPS;  // TEB to receive monitor requests
    const unsigned MEB_PORT_BASE  = MRQ_PORT_BASE + MAX_MEBS;  // MEB to receive data contributions

    const unsigned MAX_ENTRIES    = 64;                        // <= BATCH_DURATION
    const uint64_t BATCH_DURATION = MAX_ENTRIES;               // >= MAX_ENTRIES; power of 2; beam pulse ticks (1 uS)
    const unsigned MAX_LATENCY    = 1024 * 1024;               // In beam pulse ticks (1 uS)
    const unsigned MAX_BATCHES    = MAX_LATENCY / MAX_ENTRIES; // Max # of batches in circulation

    const unsigned NUM_READOUT_GROUPS = 16;     // Number of readout groups supported

    enum { WRT_IDX, MON_IDX };                  // Indexes of trigger decision results

    struct TebCtrbParams
    {
      using string_t = std::string;
      using vecstr_t = std::vector<std::string>;

      string_t ifAddr;             // Network interface to use
      string_t port;               // Served port to receive results
      unsigned partition;          // The chosen system
      string_t alias;              // Unique name passed on cmd line
      unsigned id;                 // Contributor instance identifier
      uint64_t builders;           // ID bit list of EBs
      vecstr_t addrs;              // TEB addresses
      vecstr_t ports;              // TEB ports
      size_t   maxInputSize;       // Max size of contribution
      int      core[2];            // Cores to pin threads to
      unsigned verbose;            // Level of detail to print
      uint16_t readoutGroup;       // RO group receiving trigger result data
      uint16_t contractor;         // RO group supplying trigger input  data
    };

    struct MebCtrbParams
    {
      using vecstr_t = std::vector<std::string>;

      vecstr_t addrs;              // MEB addresses
      vecstr_t ports;              // MEB ports
      unsigned id;                 // Contributor instance identifier
      unsigned maxEvents;          // Max # of events to provide for
      size_t   maxEvSize;          // Max event size
      size_t   maxTrSize;          // Max non-event size
      unsigned verbose;            // Level of detail to print
    };

    struct EbParams                // Used with both TEBs and MEBs
    {
      using string_t = std::string;
      using vecstr_t = std::vector<std::string>;
      using u64arr_t = std::array<uint64_t, NUM_READOUT_GROUPS>;

      string_t ifAddr;             // Network interface to use
      string_t ebPort;             // EB port to serve
      string_t mrqPort;            // Mon request port to receive on
      unsigned partition;          // The chosen system
      string_t alias;              // Unique name passed on cmd line
      unsigned id;                 // EB instance identifier
      uint64_t contributors;       // ID bit list of contributors
      vecstr_t addrs;              // Contributor addresses
      vecstr_t ports;              // Contributor ports
      size_t   maxTrSize;          // Max non-event Dgram size
      size_t   maxResultSize;      // Max result Dgram size
      unsigned numMrqs;            // Number of Mon request servers
      int      core[2];            // Cores to pin threads to
      unsigned verbose;            // Level of detail to print
      u64arr_t contractors;        // Ctrbs providing Inputs  per readout group
      u64arr_t receivers;          // Ctrbs expecting Results per readout group
    };
  };
};

#endif
