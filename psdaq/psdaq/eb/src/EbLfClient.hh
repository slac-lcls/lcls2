#ifndef Pds_Eb_EbLfClient_hh
#define Pds_Eb_EbLfClient_hh

#include "EbLfLink.hh"

#include <string>
#include <vector>

namespace Pds {
  namespace Eb {

    class EbLfClient
    {
    public:
      EbLfClient(const unsigned& verbose);
      EbLfClient(const unsigned&                           verbose,
                 const std::map<std::string, std::string>& kwargs);
    public:
      int connect(EbLfCltLink** link,
                  const char*   peer,
                  const char*   port,
                  unsigned      tmo);
      int disconnect(EbLfCltLink*);
    public:
      uint64_t pending() const { return _pending; }
      uint64_t posting() const { return _posting; }
    private:
      volatile uint64_t _pending;       // Flag set when currently pending
      volatile uint64_t _posting;       // Bit list of IDs currently posting
      const unsigned&   _verbose;       // Print some stuff if set
      Fabrics::Info     _info;          // Connection options
    };

    // --- Revisit: The following maybe better belongs somewhere else

    int linksConnect(EbLfClient&                     transport,
                     std::vector<EbLfCltLink*>&      links,
                     const std::vector<std::string>& addrs,
                     const std::vector<std::string>& ports,
                     unsigned                        id,
                     const char*                     name);
    int linksConfigure(std::vector<EbLfCltLink*>& links,
                       void*                      region,
                       size_t                     regSize,
                       const char*                name);
    int linksConfigure(std::vector<EbLfCltLink*>& links,
                       void*                      region,
                       size_t                     lclSize,
                       size_t                     rmtSize,
                       const char*                name);
  };
};

#endif
