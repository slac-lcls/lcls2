#ifndef Pds_Eb_MonLfClient_hh
#define Pds_Eb_MonLfClient_hh

#include <stdint.h>
#include <cstddef>
#include <string>
#include <vector>

namespace Pds {
  namespace Fabrics {

    class Endpoint;
    class MemoryRegion;

  };

  using StringList = std::vector<std::string>;
  using EpList     = std::vector<Fabrics::Endpoint*>;

  namespace Eb {

    class MonLfClient
    {
    public:
      MonLfClient(StringList& peers,
                  StringList& port);
      virtual ~MonLfClient();
    public:
      int connect(unsigned tmo);
      int post(unsigned idx, uint64_t data);
    public:
      virtual int shutdown();
    private:
      int _connect(std::string&        peer,
                   std::string&        port,
                   unsigned            tmo,
                   Fabrics::Endpoint*& ep);
    private:
      enum { scratch_size = 2 * sizeof(uint64_t) };
    private:
      StringList&            _peers;    // List of peers
      StringList&            _ports;    // The ports to listen on
      EpList                 _ep;       // Links to MonLfServers
      Fabrics::MemoryRegion* _mr;       // Shared scratch area
      char*                  _scratch;  // Scratch buffer
    };
  };
};

#endif
