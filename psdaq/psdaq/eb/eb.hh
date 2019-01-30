#ifndef Pds_Eb_Eb_hh
#define Pds_Eb_Eb_hh

#include <cstdint>
#include <string>
#include <vector>


namespace Pds {
  namespace Eb {

    const char* const PARTITION       = "Test";
    const char* const RTMON_HOST      = "psdev7b";
    const unsigned    RTMON_PORT_BASE = 55559;
    const char* const COLL_HOST       = "drp-tst-acc06";

    const unsigned MAX_DRPS        = 64;         // Maximum possible number of Contributors
    const unsigned MAX_TEBS        = 64;         // Maximum possible number of Event Builders
    const unsigned MAX_MEBS        = 64;         // Maximum possible number of Monitors

    const unsigned TEB_PORT_BASE   = 32768;                    // TEB to receive L3 contributions
    const unsigned DRP_PORT_BASE   = TEB_PORT_BASE + MAX_TEBS; // TEB to send    results
    const unsigned MRQ_PORT_BASE   = DRP_PORT_BASE + MAX_DRPS; // TEB to receive monitor requests
    const unsigned MEB_PORT_BASE   = MRQ_PORT_BASE + MAX_MEBS; // MEB to receive data contributions

    const unsigned MAX_BATCHES     = 8192;       // Maximum number of batches in circulation
    const unsigned MAX_ENTRIES     = 64;         // < or = to batch_duration
    const uint64_t BATCH_DURATION  = MAX_ENTRIES;// > or = to max_entries; power of 2; beam pulse ticks (1 uS)

    struct TebCtrbParams
    {
      std::string              ifAddr;        // Network interface to use
      std::string              port;          // Served port to receive results
      unsigned                 id;            // Contributor instance identifier
      uint64_t                 builders;      // ID bit list of EBs
      std::vector<std::string> addrs;         // TEB addresses
      std::vector<std::string> ports;         // TEB ports
      uint64_t                 duration;      // Max time duration of batches
      unsigned                 maxBatches;    // Max # of batches to provide for
      unsigned                 maxEntries;    // Max # of entries per batch
      size_t                   maxInputSize;  // Max size of contribution
      int                      core[2];       // Cores to pin threads to
      unsigned                 verbose;       // Level of detail to print
    };

    struct MebCtrbParams
    {
      std::vector<std::string> addrs;         // MEB addresses
      std::vector<std::string> ports;         // MEB ports
      unsigned                 id;            // Contributor instance identifier
      unsigned                 maxEvents;     // Junk?
      size_t                   maxEvSize;     // Max event size
      size_t                   maxTrSize;     // Max non-event size
      unsigned                 verbose;       // Level of detail to print
    };

    struct EbParams                           // Used with both TEBs and MEBs
    {
      std::string              ifAddr;        // Network interface to use
      std::string              ebPort;        // EB port to serve
      std::string              mrqPort;       // Mon request port to receive on
      unsigned                 id;            // EB instance identifier
      uint64_t                 contributors;  // ID bit list of contributors
      std::vector<std::string> addrs;         // Contributor addresses
      std::vector<std::string> ports;         // Contributor ports
      uint64_t                 duration;      // Max time duration of buffers
      unsigned                 maxBuffers;    // Max # of buffers to provide for
      unsigned                 maxEntries;    // Max # of entries per buffer
      unsigned                 numMrqs;       // Number of Mon request servers
      size_t                   maxTrSize;     // Max non-event Dgram size
      size_t                   maxResultSize; // Max result Dgram size
      int                      core[2];       // Cores to pin threads to
      unsigned                 verbose;       // Level of detail to print
    };
  };
};

#endif
