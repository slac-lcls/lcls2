#ifndef Pds_Eb_EbFtBase_hh
#define Pds_Eb_EbFtBase_hh

#include <stdint.h>
#include <cstddef>
#include <vector>


namespace Pds {

  namespace Fabrics {

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
      EbFtBase(unsigned nPeers);
      virtual ~EbFtBase();
    public:
      Fabrics::MemoryRegion* registerMemory(void* buffer, size_t size);
      uint64_t pend();
      int      post(Fabrics::LocalIOVec&, size_t len, unsigned dst, uint64_t offset, void* ctx);
      uint64_t rmtAdx(unsigned dst, uint64_t offset); // Revisit: For debugging, remove
    public:
      virtual int shutdown() = 0;
    protected:
      void     _mapIds(unsigned nPeers);
      int      _syncLclMr(char*                   region,
                          size_t                  size,
                          Fabrics::Endpoint*      ep,
                          Fabrics::MemoryRegion*  mr,
                          Fabrics::RemoteAddress& ra);
      int      _syncRmtMr(char*                   region,
                          size_t                  size,
                          Fabrics::Endpoint*      ep,
                          Fabrics::MemoryRegion*  mr,
                          Fabrics::RemoteAddress& ra);
    private:
      uint64_t _tryCq();
    protected:
      EpList                     _ep;   // List of Endpoints
      MrList                     _mr;   // List of Memory Regions per EP
      RaList                     _ra;   // List of remote address descriptors
      Fabrics::CompletionPoller* _cqPoller;
      std::vector<unsigned>      _id;
      unsigned*                  _mappedId;
    };
  };
};

#endif
