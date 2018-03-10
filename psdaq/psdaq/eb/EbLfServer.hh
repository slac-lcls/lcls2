#ifndef Pds_Eb_EbLfServer_hh
#define Pds_Eb_EbLfServer_hh

#include "EbLfBase.hh"

#include <stdint.h>
#include <cstddef>
#include <string>
#include <vector>

namespace Pds {
  namespace Fabrics {

    class PassiveEndpoint;
    class Endpoint;

  };

  namespace Eb {

    class EbLfServer : public EbLfBase
    {
    public:
      EbLfServer(const char*  addr,
                 std::string& port,
                 unsigned     nClients);
      virtual ~EbLfServer();
    public:
      int connect(unsigned id);
    public:
      virtual int shutdown();
    private:
      int _connect();
      int _exchangeIds(Fabrics::Endpoint* ep,
                       unsigned           myId,
                       unsigned&          id);
    private:
      const char*               _addr; // The interface address to use
      std::string&              _port; // The port to listen on
      Fabrics::PassiveEndpoint* _pep;  // Endpoint for establishing connections
    };
  };
};

#endif
