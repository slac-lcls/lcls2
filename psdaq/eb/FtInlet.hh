#ifndef Pds_FtInlet_hh
#define Pds_FtInlet_hh

#include <stdint.h>
#include <cstddef>
#include <string>
#include <vector>

namespace Pds {

  namespace Fabrics {

    class PassiveEndpoint;
    class Endpoint;
    class MemoryRegion;

  };

#define EpList std::vector<Fabrics::Endpoint*>
#define MrList std::vector<Fabrics::MemoryRegion*>

  class FtInlet
  {
  public:
    FtInlet(std::string& port);
    ~FtInlet();
  public:
    int   connect(char* base, unsigned nPeers, size_t size, bool shared);
    void* pend();
    int   shutdown();
  private:
    std::string&              _port;   // The port to listen on
    Fabrics::PassiveEndpoint* _pep;    // Endpoint for establishing connections
    EpList                    _ep;     // List of Endpoints
    MrList                    _mr;     // List of Memory Regions per EP
    //Fabrics::MemoryRegion*    _mr;
  };
};

#endif
