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

    class EbFtBase;

    class EbFtStats
    {
    public:
      EbFtStats(unsigned nPeers);
      ~EbFtStats();
    public:
      void clear();
      void dump();
    private:
      friend class EbFtBase;
    private:
      uint64_t              _postCnt;
      uint64_t              _repostCnt;
      uint64_t              _repostMax;
      uint64_t              _postWtAgnCnt;
      uint64_t              _pendCnt;
      uint64_t              _pendTmoCnt;
      uint64_t              _rependCnt;
      uint64_t              _rependMax;
      std::vector<uint64_t> _rmtWrCnt;
      std::vector<uint64_t> _compAgnCnt;
    };

    class EbFtBase
    {
    public:
      EbFtBase(unsigned nPeers);
      virtual ~EbFtBase();
    public:
      void     registerMemory(void* buffer, size_t size);
      int      pend(uint64_t* data);
      //int      post(Fabrics::LocalIOVec&, size_t len, unsigned dst, uint64_t offset, void* ctx);
      int      post(const void* buf, size_t len, unsigned dst, uint64_t offset);
      uint64_t rmtAdx(unsigned dst, uint64_t offset); // Revisit: For debugging, remove
    public:
      virtual int shutdown() = 0;
    public:
      const EbFtStats& stats() const;
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
      int      _tryCq(uint64_t* data);
    protected:
      EpList                     _ep;   // List of Endpoints
      MrList                     _lMr;  // List of local  Memory Regions per EP
      MrList                     _rMr;  // List of remote Memory Regions per EP
      RaList                     _ra;   // List of remote address descriptors
      Fabrics::CompletionPoller* _cqPoller;
      std::vector<unsigned>      _id;
      unsigned*                  _mappedId;
      EbFtStats                  _stats;
    private:
      unsigned                   _iSrc;
    };
  };
};

#endif
