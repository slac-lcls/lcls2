#ifndef Pds_Eb_MebContributor_hh
#define Pds_Eb_MebContributor_hh

#include "psdaq/eb/eb.hh"

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

    class MebContributor
    {
    public:
      MebContributor(const MebCtrbParams& prms);
      ~MebContributor();
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
      size_t               _maxEvSize;
      size_t               _maxTrSize;
      std::vector<size_t>  _trOffset;
      void*                _region;
      Pds::Eb::EbLfClient* _transport;
      EbLfLinkMap          _links;
      const unsigned       _id;
      bool                 _verbose;
    private:
      uint64_t             _eventCount;
    };
  };
};

#endif
