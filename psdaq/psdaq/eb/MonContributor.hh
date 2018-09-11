#ifndef Pds_Eb_MonContributor_hh
#define Pds_Eb_MonContributor_hh

#include <cstdint>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>
#include <atomic>


namespace XtcData {
  class Dgram;
};

namespace Pds {
  namespace Eb {

    class EbLfLink;
    class EbLfClient;

    using EbLfLinkMap = std::unordered_map<unsigned, Pds::Eb::EbLfLink*>;

    struct MonCtrbParams
    {
      std::vector<std::string> addrs;
      std::vector<std::string> ports;
      unsigned                 id;
      unsigned                 maxEvents;
      size_t                   maxEvSize;
      size_t                   maxTrSize;
      unsigned                 verbose;
    };

    class MonContributor
    {
    public:
      MonContributor(const MonCtrbParams& prms);
      ~MonContributor();
    public:
      int      post(const XtcData::Dgram* dataDatagram); // Transitions
      int      post(const XtcData::Dgram* dataDatagram,
                    uint32_t              destination);  // L1Accepts
    public:
      const uint64_t& eventCount() { return _eventCount; }
    private:
      void    _initialize(const char*                     who,
                          const std::vector<std::string>& addrs,
                          const std::vector<std::string>& ports,
                          unsigned                        id,
                          size_t                          regionSize);
    private:
      size_t                                 _maxEvSize;
      size_t                                 _maxTrSize;
      std::vector<size_t>                    _trOffset;
      void*                                  _region;
      Pds::Eb::EbLfClient*                   _transport;
      //std::vector<Pds::Eb::EbLfLink*>      _links;
      //std::unordered_map<unsigned, unsigned> _id2Idx;
      EbLfLinkMap                            _links;
      const unsigned                         _id;
      bool                                   _verbose;
    private:
      uint64_t                               _eventCount;
    };
  };
};

#endif
