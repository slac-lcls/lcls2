#ifndef Pds_Eb_EbLfClient_hh
#define Pds_Eb_EbLfClient_hh

#include "EbLfBase.hh"

#include <stdint.h>
#include <cstddef>
#include <string>
#include <vector>

namespace Pds {
  namespace Fabrics {

    class Endpoint;

  };

  typedef std::vector<std::string> StringList;

  namespace Eb {

    class EbLfClient : public EbLfBase
    {
    public:
      EbLfClient(StringList& peers,
                 StringList& port);
      virtual ~EbLfClient();
    public:
      int connect(unsigned    id,
                  unsigned    tmo,
                  void*       region,
                  size_t      size,
                  PeerSharing shared = EbLfBase::PEERS_SHARE_BUFFERS,
                  void*       ctx    = nullptr);
    public:
      virtual int shutdown();
    private:
      int _connect    (std::string&               peer,
                       std::string&               port,
                       unsigned                   tmo,
                       Fabrics::Endpoint*&        ep,
                       Fabrics::CompletionQueue*& txcq);
      int _exchangeIds(Fabrics::Endpoint*         ep,
                       Fabrics::MemoryRegion*     mr,
                       unsigned                   myId,
                       unsigned&                  id);
    private:
      StringList& _peers;               // List of peers
      StringList& _ports;               // The ports to listen on
    };
  };
};

#endif
