#ifndef Pds_Eb_EbCtrbInBase_hh
#define Pds_Eb_EbCtrbInBase_hh

#include "EbLfServer.hh"

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
      EbCtrbInBase(const TebCtrbParams&, const std::shared_ptr<MetricExporter>&);
      virtual ~EbCtrbInBase();
    public:
      int      resetCounters();
      int      startConnection(std::string& port);
      int      connect(size_t resSizeGuess, unsigned numBuffers);
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
      int     _linksConfigure(std::vector<EbLfSvrLink*>& links,
                              unsigned                   id,
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
    private:
      EbLfServer                    _transport;
      std::vector<EbLfSvrLink*>     _links;
      size_t                        _maxResultSize;
      const EbDgram*                _inputs;
      std::list<const ResultDgram*> _deferred;
      uint64_t                      _batchCount;
      uint64_t                      _eventCount;
      uint64_t                      _missing;
      uint64_t                      _bypassCount;
      uint64_t                      _noProgCount;
      uint64_t                      _prvNPCnt;
      const TebCtrbParams&          _prms;
      size_t                        _regSize;
      void*                         _region;
    };
  };
};

#endif
