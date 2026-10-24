/**
 * @file
 * @brief EbLfClient, which opens client-side libfabric links (EbLfCltLink) to event-builder servers, and helpers that connect and configure a set of such links.
 */
#ifndef Pds_Eb_EbLfClient_hh
#define Pds_Eb_EbLfClient_hh

#include "EbLfLink.hh"

#include <string>
#include <vector>

namespace Pds {
  namespace Eb {

    /** Creates and tears down client-side EbLfCltLink connections; all its links share one pending word and one posting word. */
    class EbLfClient
    {
    public:
      /** Construct with default libfabric hints (Info) and zeroed pending and posting words; verbose is kept by reference. */
      EbLfClient(const unsigned& verbose);
      /** Construct with libfabric hints built from kwargs (ep_fabric, ep_domain, ep_provider are used by Info) and zeroed pending and posting words. */
      EbLfClient(const unsigned&                           verbose,
                 const std::map<std::string, std::string>& kwargs);
    public:
      /** Create a Fabric, a transmit completion queue and an Endpoint for peer:port and try to connect, retrying refused connections every 1 ms until tmo milliseconds have passed (tmo 0 means no limit). On success store a new EbLfCltLink in link and return 0; otherwise return a negative error (-FI_EAGAIN on timeout). */
      int connect(EbLfCltLink** link,
                  const char*   peer,
                  const char*   port,
                  unsigned      tmo);
      /** Delete link and its endpoint, the endpoint's transmit completion queue and its fabric; does nothing for a null link. Always returns FI_SUCCESS. */
      int disconnect(EbLfCltLink*);
    public:
      /** Return the shared pending word, which EbLfLink::poll(data, msTmo) increments while it waits. */
      uint64_t pending() const { return _pending; }
      /** Return the shared posting word, a bit list of the peer IDs whose links are currently posting. */
      uint64_t posting() const { return _posting; }
    private:
      volatile uint64_t _pending;       // Flag set when currently pending
      volatile uint64_t _posting;       // Bit list of IDs currently posting
      const unsigned&   _verbose;       // Print some stuff if set
      Fabrics::Info     _info;          // Connection options
    };

    // --- Revisit: The following maybe better belongs somewhere else

    /** Connect transport to every addrs[i]:ports[i] (14750 ms timeout each), then exchange IDs on each link and store it in links at the index of the remote ID. name labels the peers in messages. Returns 0 or the first error. */
    int linksConnect(EbLfClient&                     transport,
                     std::vector<EbLfCltLink*>&      links,
                     const std::vector<std::string>& addrs,
                     const std::vector<std::string>& ports,
                     unsigned                        id,
                     const char*                     name);
    /** Call linksConfigure(links, region, regSize, regSize, name). */
    int linksConfigure(std::vector<EbLfCltLink*>& links,
                       void*                      region,
                       size_t                     regSize,
                       const char*                name);
    /** Call EbLfCltLink::prepare(region, lclSize, rmtSize, name) on every link and log the configured region. Returns 0 or the first error. */
    int linksConfigure(std::vector<EbLfCltLink*>& links,
                       void*                      region,
                       size_t                     lclSize,
                       size_t                     rmtSize,
                       const char*                name);
  };
};

#endif
