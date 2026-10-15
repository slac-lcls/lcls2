#ifndef Pds_Eb_EbCtrbInBase_hh
#define Pds_Eb_EbCtrbInBase_hh

#include "EbLfServer.hh"
#include "eb.hh"                        // MAX_TEBS

#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <list>


namespace Pds
{
  class MetricExporter;
  class EbDgram;

  namespace Eb
  {
    class TebCtrbParams;
    class EbLfSvrLink;
    class TebContributor;
    class ResultDgram;

    class EbCtrbInBase
    {
    public:
      EbCtrbInBase(const TebCtrbParams&);
      virtual ~EbCtrbInBase();
    public:
      int      resetCounters();
      int      startConnection(std::string& port);
      int      connect(const std::shared_ptr<MetricExporter>);
      int      configure(unsigned numBuffers);
      void     unconfigure();
      void     disconnect();
      void     shutdown();
    public:
      void     receiver(TebContributor&, std::atomic<bool>& running);
    public:
      virtual
      void     process(const ResultDgram& result, unsigned index) = 0;
    private:
      int     _setupMetrics(const std::shared_ptr<MetricExporter>);
      int     _linksConfigure(std::vector<EbLfSvrLink*>& links,
                              unsigned                   numBuffers,
                              const char*                name);
      int     _process(TebContributor& ctrb);
      void    _matchUp(TebContributor&    ctrb,
                       const ResultDgram* results);
      void    _defer(const ResultDgram* results);
      void    _deliverBypass(TebContributor& ctrb,
                             const EbDgram*& inputs);
      void    _deliver(TebContributor&     ctrb,
                       const ResultDgram*& results,
                       const EbDgram*&     inputs);
      void    _dump(TebContributor&    ctrb,
                    const ResultDgram* results,
                    const EbDgram*     inputs) const;
      void    _slotDump();
      bool    _orphaned(uint64_t iPid) const;
      void    _orphanDump(TebContributor& ctrb, const EbDgram* inp);
    private:
      EbLfServer                    _transport;
      std::vector<EbLfSvrLink*>     _links;
      unsigned                      _numBuffers;
      size_t                        _maxResultSize;
      const EbDgram*                _inputs;
      std::list<const ResultDgram*> _deferred;
      uint64_t                      _batchCount;
      uint64_t                      _eventCount;
      uint64_t                      _missing;
      uint64_t                      _bypassCount;
      uint64_t                      _noProgCount;
      uint64_t                      _prvNPCnt;
      // Newest batch-head pulse ID received per TEB, and a count of batches
      // from each.  A Result older than every TEB's newest can no longer be
      // in flight, the transport delivering writes to one region in order.
      uint64_t                      _newestRcvd[MAX_TEBS];
      uint64_t                      _rcvdFrom[MAX_TEBS];
      uint64_t                      _orphanCount;
      const TebCtrbParams&          _prms;
      size_t                        _regSize;
      void*                         _region;
    };
  };
};

#endif
