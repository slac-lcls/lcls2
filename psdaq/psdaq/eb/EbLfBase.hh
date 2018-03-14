#ifndef Pds_Eb_EbLfBase_hh
#define Pds_Eb_EbLfBase_hh

#include <stdint.h>
#include <cstddef>
#include <vector>


struct fi_cq_data_entry;

namespace Pds {

  namespace Fabrics {

    class Endpoint;
    class MemoryRegion;
    class RemoteAddress;
    class LocalIOVec;
    class CompletionPoller;
  };

  typedef std::vector<Fabrics::Endpoint*>     EpList;
  typedef std::vector<Fabrics::MemoryRegion*> MrList;
  typedef std::vector<Fabrics::RemoteAddress> RaList;

  namespace Eb {

    class EbLfBase;

    class EbLfStats
    {
    public:
      EbLfStats(unsigned nPeers);
      ~EbLfStats();
    public:
      void clear();
      void dump();
    private:
      friend class EbLfBase;
    private:
      uint64_t              _postCnt;
      uint64_t              _repostCnt;
      uint64_t              _repostMax;
      uint64_t              _pendCnt;
      uint64_t              _pendTmoCnt;
      uint64_t              _rependCnt;
      uint64_t              _rependMax;
      std::vector<uint64_t> _rmtWrCnt;
      std::vector<uint64_t> _compAgnCnt;
    };

    class EbLfBase
    {
    public:
      enum PeerSharing { PER_PEER_BUFFERS, PEERS_SHARE_BUFFERS };
    public:
      EbLfBase(unsigned nPeers);
      virtual ~EbLfBase();
    public:
      const char* base() const;
      int         prepareRmtMr(void* buffer, size_t rmtSize);
      int         prepareLclMr(size_t lclSize, PeerSharing shared = PER_PEER_BUFFERS);
      int         postCompRecv(unsigned dst, unsigned count, void* ctx);
      int         pend(fi_cq_data_entry*);
      int         pendW(fi_cq_data_entry*);
      int         post(unsigned dst, const void* buf, size_t len, uint64_t os, uint64_t immData, void* ctx = nullptr);
      //int         post(Fabrics::LocalIOVec&, size_t len, unsigned dst, uint64_t offset, void* ctx);
      void*       lclAdx(unsigned src, uint64_t offset) const;
      uintptr_t   rmtAdx(unsigned dst, uint64_t offset) const; // Revisit: For debugging, remove
    public:
      virtual int shutdown() = 0;
    public:
      const EbLfStats& stats() const;
    protected:
      void     _mapIds(unsigned nPeers);
      //int      _postCompRecv(Fabrics::Endpoint* ep, unsigned count);
    private:
      int      _tryCq(fi_cq_data_entry*);
    protected:
      EpList                     _ep;   // List of Endpoints
      MrList                     _lMr;  // List of local  Memory Regions per EP
      MrList                     _rMr;  // List of remote Memory Regions per EP
      RaList                     _ra;   // List of remote address descriptors
      Fabrics::CompletionPoller* _cqPoller;
      char*                      _base; // Aligned local memory region
      unsigned                   _rxDepth;
      std::vector<unsigned>      _rOuts;
      std::vector<unsigned>      _id;
      unsigned*                  _mappedId;
      EbLfStats                  _stats;
    private:
      unsigned                   _iSrc;
    };
  };
};

#endif
