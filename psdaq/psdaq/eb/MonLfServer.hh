#ifndef Pds_Eb_MonLfServer_hh
#define Pds_Eb_MonLfServer_hh

#include <stdint.h>
#include <cstddef>
#include <string>
#include <vector>
#include <atomic>
#include <thread>

namespace Pds {
  namespace Fabrics {

    class PassiveEndpoint;
    class MemoryRegion;
    class CompletionQueue;
    class Endpoint;

  };

  namespace Eb {

    using EpList = std::vector<Fabrics::Endpoint*>;

    class MonLfServer
    {
    public:
      MonLfServer(const char*  addr,
                  std::string& port);
      virtual ~MonLfServer();
    public:
      int connect();
      int postCompRecv(unsigned dst, void* ctx=NULL);
      int poll(uint64_t* data);
    public:
      virtual int shutdown();
    private:
      int  _postCompRecv(Fabrics::Endpoint*, unsigned count, void* ctx=NULL);
      void _listen();
    private:
      enum { scratch_size = 2 * sizeof(uint64_t) };
    private:
      const char*               _addr;    // The interface address to use
      std::string&              _port;    // The port to listen on
      Fabrics::PassiveEndpoint* _pep;     // EP for establishing connections
      Fabrics::MemoryRegion*    _mr;      // Shared scratch area
      Fabrics::CompletionQueue* _rxcq;    // CQ for watching for EP activity
      int                       _rxDepth;
      std::vector<int>          _rOuts;
      char*                     _scratch; // Scratch buffer
      EpList                    _ep;      // Links to MonLfClients
      std::atomic<bool>         _running;
      std::thread*              _listener;
    };
  };
};

#endif
