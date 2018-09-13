#ifndef Pds_Eb_Eb_hh
#define Pds_Eb_Eb_hh

#include <cstdint>
#include <string>
#include <vector>


namespace Pds {
  namespace Eb {

    const unsigned MAX_DRPS       = 64;         // Maximum possible number of Contributors
    const unsigned MAX_TEBS       = 64;         // Maximum possible number of Event Builders
    const unsigned MAX_MEBS       = 64;         // Maximum possible number of Monitors

    const unsigned TEB_PORT_BASE  = 32768;                    // TEB to receive L3 contributions
    const unsigned DRP_PORT_BASE  = TEB_PORT_BASE + MAX_TEBS; // TEB to send    results
    const unsigned MRQ_PORT_BASE  = DRP_PORT_BASE + MAX_DRPS; // TEB to receive monitor requests
    const unsigned MEB_PORT_BASE  = MRQ_PORT_BASE + MAX_MEBS; // MEB to receive data contributions

    const unsigned MAX_BATCHES    = 8192;       // Maximum number of batches in circulation
    const unsigned MAX_ENTRIES    = 64;         // < or = to batch_duration
    const uint64_t BATCH_DURATION = MAX_ENTRIES;// > or = to max_entries; power of 2; beam pulse ticks (1 uS)

    struct EbCtrbParams
    {
      std::vector<std::string> addrs;
      std::vector<std::string> ports;
      char*                    ifAddr;
      std::string              port;
      unsigned                 id;
      uint64_t                 builders;
      uint64_t                 duration;
      unsigned                 maxBatches;
      unsigned                 maxEntries;
      size_t                   maxInputSize;
      size_t                   maxResultSize;
      int                      core[2];
      unsigned                 verbose;
    };

    struct MonCtrbParams
    {
      std::vector<std::string> addrs;
      std::vector<std::string> ports;
      unsigned                 id;
      unsigned                 maxEvents;
      size_t                   maxEvSize;
      size_t                   maxTrSize;
      unsigned                 verbose;
    };
  };
};

#endif
