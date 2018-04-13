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
    class CompletionQueue;
    class CompletionPoller;
  };

  typedef std::vector<Fabrics::Endpoint*>        EpList;
  typedef std::vector<Fabrics::MemoryRegion*>    MrList;
  typedef std::vector<Fabrics::RemoteAddress>    RaList;
  typedef std::vector<Fabrics::CompletionQueue*> CqList;

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
      uint64_t _postCnt;
      uint64_t _repostCnt;
      uint64_t _repostMax;
      uint64_t _pendCnt;
      uint64_t _pendTmoCnt;
      uint64_t _rependCnt;
      uint64_t _rependMax;
    };

    class EbLfBase
    {
    public:
      enum PeerSharing { PER_PEER_BUFFERS, PEERS_SHARE_BUFFERS };
    public:
      EbLfBase(unsigned nPeers);
      virtual ~EbLfBase();
    public:
      void*       lclAdx(unsigned src, uint64_t offset) const;
      uintptr_t   rmtAdx(unsigned dst, uint64_t offset) const;
      int         postCompRecv(unsigned dst, void* ctx=NULL);
      int         pend(fi_cq_data_entry*);
      int         post(unsigned dst, const void* buf, size_t len, uint64_t offset, uint64_t immData, void* ctx = nullptr);
    public:
      virtual int shutdown() = 0;
    public:
      const EbLfStats& stats() const;
    protected:
      int      _setupMr(Fabrics::Endpoint*        ep,
                        void*                     region,
                        size_t                    size,
                        Fabrics::MemoryRegion*&   mr);
      int      _syncLclMr(Fabrics::Endpoint*      ep,
                          Fabrics::MemoryRegion*  mr,
                          Fabrics::RemoteAddress& ra);
      int      _syncRmtMr(Fabrics::Endpoint*      ep,
                          Fabrics::MemoryRegion*  mr,
                          Fabrics::RemoteAddress& ra,
                          size_t                  size);
      int      _postCompRecv(Fabrics::Endpoint*, unsigned count, void* ctx=NULL);
      void     _mapIds(unsigned nPeers);
    private:
      int      _tryCq(fi_cq_data_entry*);
    protected:
      EpList                     _ep;   // List of Endpoints
      MrList                     _mr;   // List of Memory Regions per EP
      RaList                     _ra;   // List of remote address descriptors
      CqList                     _txcq;
      Fabrics::CompletionQueue*  _rxcq;
      int                        _rxDepth;
      std::vector<int>           _rOuts;
      std::vector<unsigned>      _id;
      unsigned*                  _mappedId;
      EbLfStats                  _stats;
    };
  };
};

#endif
