#ifndef Pds_FtInlet_hh
#define Pds_FtInlet_hh

#include <vector>

namespace Pds {

  class Fabrics::Endpoint;
  class Fabrics::MemoryRegion;

#define EpList std::vector<Fabrics::Endpoint*>
#define MrList std::vector<Fabrics::MemoryRegion*>

  class FtInlet : public XptInlet
  {
  public:
    FtInlet(std::string& port);
    ~FtInlet();
  public:
    int  connect(unsigned nPeers, unsigned nSlots, size_t size, char* base);
    void pend();
    void shutdown();
  private:
    std::string&              _port     //
    Fabrics::PassiveEndpoint* _pep;     //
    EpList&                   _ep;      // List of Endpoints
    MrList&                   _mr;      // List of Memory Regions per EP
  };
};

#endif
