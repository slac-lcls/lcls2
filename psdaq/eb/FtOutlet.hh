#ifndef Pds_FtOutlet_hh
#define Pds_FtOutlet_hh

#include <stdint.h>
#include <cstddef>
#include <string>
#include <vector>

struct fi_msg_rma;

namespace Pds {

  namespace Fabrics {

    class Endpoint;
    class MemoryRegion;
    class RemoteAddress;

  };

// Notational conveniences that can't apparently be typedefed...
#define StringList std::vector<std::string>
#define EpList     std::vector<Fabrics::Endpoint*>
#define MrList     std::vector<Fabrics::MemoryRegion*>
#define RaList     std::vector<Fabrics::RemoteAddress>

  class FtOutlet
  {
  public:
    FtOutlet(StringList&  remote,
             std::string& port);
    ~FtOutlet();
  public:
    int connect(size_t size, unsigned tmo);
    int post(fi_msg_rma*, unsigned dst, uint64_t dstOffset, void* ctx);
    int shutdown();
  private:
    int _connect(std::string&            remote,
                 std::string&            port,
                 size_t                  size,
                 Fabrics::Endpoint*&     ep,
                 Fabrics::MemoryRegion*& mr,
                 Fabrics::RemoteAddress& ra,
                 unsigned                tmo);
  private:
    StringList&  _remote;               // List of peers
    std::string& _port;                 // The port to listen on
    EpList       _ep;                   // List of endpoints
    MrList       _mr;                   // List of memory regions
    RaList       _ra;                   // List of remote address descriptors
  private:
    char*        _scratch;              // Internal scratch pool
  };
};

#endif
