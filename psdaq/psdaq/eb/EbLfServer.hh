#ifndef Pds_Eb_EbLfServer_hh
#define Pds_Eb_EbLfServer_hh

#include "EbLfLink.hh"

#include <stdint.h>
#include <cstddef>
#include <string>


struct fi_cq_data_entry;

namespace Pds {

  namespace Fabrics {
    class PassiveEndpoint;
    class CompletionQueue;
  };

  namespace Eb {

    class EbLfServer
    {
    public:
      EbLfServer(unsigned verbose);
    public:
      int initialize(const std::string& addr, const std::string& port);
      int connect(EbLfLink**, int msTmo = -1);
      int pend(fi_cq_data_entry*, int msTmo);
      int pend(void** context, int msTmo);
      int pend(uint64_t* data, int msTmo);
      int poll(uint64_t* data);
    public:
      const uint64_t& pending() const { return _pending; }
    public:
      int shutdown(EbLfLink*);
    private:
      int _poll(fi_cq_data_entry*, uint64_t flags);
    private:                            // Arranged in order of access frequency
      Fabrics::CompletionQueue* _rxcq;    // Receive completion queue
      int                       _tmo;     // Timeout for polling or waiting
      unsigned                  _verbose; // Print some stuff if set
    private:
      uint64_t                  _pending; // Flag set when currently pending
      uint64_t                  _unused;  // Bit list of IDs currently posting
    private:
      Fabrics::PassiveEndpoint* _pep;   // Endpoint for establishing connections
    };
  };
};

#endif
