#ifndef Pds_Eb_EbAppBase_hh
#define Pds_Eb_EbAppBase_hh

#include "eb.hh"
#include "EventBuilder.hh"
#include "EbLfServer.hh"

#include <cstdint>
#include <cstddef>
#include <string>
#include <array>
#include <vector>


namespace XtcData {
  class Dgram;
  class TimeStamp;
};

namespace Pds {
  namespace Eb {

    using u64arr_t = std::array<uint64_t, NUM_READOUT_GROUPS>;

    class EbLfSvrLink;
    class EbEvent;

    class EbAppBase : public EventBuilder
    {
    public:
      EbAppBase(const EbParams& prms,
                const uint64_t  duration,
                const unsigned  maxEntries,
                const unsigned  maxBuffers);
      virtual ~EbAppBase() {}
    public:
      const uint64_t&  rxPending() const { return _transport.pending(); }
      const uint64_t&  bufferCnt() const { return _bufferCnt; }
      const uint64_t&  fixupCnt()  const { return _fixupCnt; }
      int              checkEQ()  { return _transport.pollEQ(); }
    public:
      int              configure(const EbParams&);
      void             shutdown();
      int              process();
      void             trim(unsigned dst);
    public:                            // For EventBuilder
      virtual void     fixup(Pds::Eb::EbEvent* event, unsigned srcId);
      virtual uint64_t contract(const XtcData::Dgram* contrib) const;
    private:                           // Arranged in order of access frequency
      u64arr_t                  _contract;
      Pds::Eb::EbLfServer       _transport;
      std::vector<EbLfSvrLink*> _links;
      std::vector<size_t>       _trRegSize;
      std::vector<size_t>       _maxTrSize;
      std::vector<size_t>       _maxBufSize;
      const unsigned            _maxEntries;
      const unsigned            _maxBuffers;
      //EbDummyTC                 _dummy;   // Template for TC of dummy contributions  // Revisit: ???
      const unsigned            _verbose;
      uint64_t                  _bufferCnt;
      uint64_t                  _fixupCnt;
    private:
      void*                     _region;
      uint64_t                  _contributors;
      unsigned                  _id;
    };
  };
};

#endif

