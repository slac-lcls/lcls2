#ifndef Pds_Eb_MebContributor_hh
#define Pds_Eb_MebContributor_hh

#include "eb.hh"
#include "EbLfClient.hh"

#include <cstdint>
#include <memory>
#include <vector>


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
      int  connect(const MebCtrbParams& prms, void* region, size_t size);
      int  configure(void* region, size_t size);
      void unconfigure();
      void disconnect();
      void shutdown();
      bool enabled() { return _enabled; }
    public:
      int  post(const Pds::EbDgram* dataDatagram); // Transitions
      int  post(const Pds::EbDgram* dataDatagram,
                uint32_t            destination);  // L1Accepts
    private:
      size_t                    _maxEvSize;
      size_t                    _maxTrSize;
      size_t                    _bufRegSize;
      EbLfClient                _transport;
      std::vector<EbLfCltLink*> _links;
      bool                      _enabled;
      unsigned                  _id;
      unsigned                  _verbose;
    private:
      uint64_t                  _eventCount;
    };
  };
};

#endif
