#ifndef Pds_Eb_EbFtServer_hh
#define Pds_Eb_EbFtServer_hh

#include "EbFtBase.hh"

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

  namespace Eb {

    class EbFtServer : public EbFtBase
    {
    public:
      EbFtServer(std::string& port,
                 unsigned     nClients,
                 size_t       lclSize);
      virtual ~EbFtServer();
    public:
      int connect();
    public:
      virtual int shutdown();
    private:
      std::string&              _port;   // The port to listen on
      Fabrics::PassiveEndpoint* _pep;    // Endpoint for establishing connections
      size_t                    _lclSize;// Local  memory region size
      char*                     _base;   // The local memory region
    };
  };
};

#endif
