#ifndef Pds_Eb_MebContributor_hh
#define Pds_Eb_MebContributor_hh

#include "eb.hh"
#include "EbLfClient.hh"

#include <cstdint>
#include <memory>
#include <vector>
#include <list>


namespace Pds {
  class EbDgram;
  class MetricExporter;

  namespace Eb {

    class EbLfCltLink;

    class MebContributor
    {
    public:
      MebContributor(const MebCtrbParams&, std::shared_ptr<MetricExporter>);
    public:
      int  resetCounters();
      int  connect();
      int  configure();
      void unconfigure();
      void disconnect();
      void shutdown();
      bool enabled() { return _enabled; }
    public:
      int  post(const Pds::EbDgram* dataDatagram); // Transitions
      int  post(const Pds::EbDgram* dataDatagram,
                uint32_t            destination);  // L1Accepts
    private:
      int _linksConfigure(const MebCtrbParams&       prms,
                          std::vector<EbLfCltLink*>& links,
                          const char*                peer);
    public:
      using listU32_t = std::list<uint32_t>;
    private:
      const MebCtrbParams&      _prms;
      size_t                    _maxEvSize;
      size_t                    _maxTrSize;
      EbLfClient                _transport;
      std::vector<EbLfCltLink*> _links;
      std::vector<void*>        _region;
      std::vector<size_t>       _regSize;
      std::vector<size_t>       _bufRegSize;
      std::vector<listU32_t>    _trBuffers;
      unsigned                  _id;
      bool                      _enabled;
      unsigned                  _verbose;
      uint64_t                  _previousPid;
    private:
      uint64_t                  _eventCount;
      uint64_t                  _trCount;
    };
  };
};

#endif
