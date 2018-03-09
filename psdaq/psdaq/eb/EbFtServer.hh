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
      enum PeerSharing { PER_PEER_BUFFERS, PEERS_SHARE_BUFFERS };
    public:
      EbFtServer(const char*  addr,
                 std::string& port,
                 unsigned     nClients,
                 size_t       lclSize,
                 PeerSharing  shared);
      virtual ~EbFtServer();
    public:
      int connect(unsigned id);
      const char* base() const;
    public:
      virtual int shutdown();
    private:
      int _connect();
      int _exchangeIds(unsigned                 myId,
                       char*                    pool,
                       size_t                   lclSize,
                       Fabrics::Endpoint*       ep,
                       Fabrics::MemoryRegion*&  mr,
                       unsigned&                id);
    private:
      const char*               _addr;   // The interface address to use
      std::string&              _port;   // The port to listen on
      Fabrics::PassiveEndpoint* _pep;    // Endpoint for establishing connections
      size_t                    _lclSize;// Local  memory region size
      const bool                _shared; // True when buffers are shared by peers
      char*                     _lclMem; // The local memory region
      char*                     _base;   // Aligned local memory region
    };
  };
};

#endif
