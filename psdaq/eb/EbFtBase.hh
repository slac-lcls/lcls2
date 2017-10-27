#ifndef Pds_Eb_EbFtBase_hh
#define Pds_Eb_EbFtBase_hh

#include <stdint.h>
#include <cstddef>
#include <vector>


namespace Pds {

  namespace Fabrics {

    class PassiveEndpoint;
    class Endpoint;
    class MemoryRegion;
    class RemoteAddress;
    class LocalIOVec;
    class CompletionPoller;
  };

#define EpList std::vector<Fabrics::Endpoint*>
#define MrList std::vector<Fabrics::MemoryRegion*>
#define RaList std::vector<Fabrics::RemoteAddress>

  namespace Eb {

    class EbFtBase
    {
    public:
      EbFtBase(unsigned nClients);
      virtual ~EbFtBase();
    public:
      uint64_t pend();
      int      post(Fabrics::LocalIOVec&, size_t len, unsigned dst, uint64_t dstOffset, void* ctx);
    public:
      virtual int shutdown() = 0;
    protected:
      int      _syncLclMr(char*                   region,
                          size_t                  size,
                          Fabrics::Endpoint*      ep,
                          Fabrics::MemoryRegion*& mr);
      int      _syncRmtMr(char*                   region,
                          size_t                  size,
                          Fabrics::Endpoint*      ep,
                          Fabrics::MemoryRegion*  mr,
                          Fabrics::RemoteAddress& ra);
    private:
      uint64_t _tryCq();
    protected:
      EpList   _ep;                     // List of Endpoints
      MrList   _mr;                     // List of Memory Regions per EP
      RaList   _ra;                     // List of remote address descriptors
      Fabrics::CompletionPoller* _cqPoller;
    private:
      unsigned _iSrc;
    };
  };
};

#endif
