#ifndef Pds_CubeResult_hh
#define Pds_CubeResult_hh

#include "psdaq/eb/src/ResultDgram.hh"
#include <vector>

namespace Drp {

    class CubeResult {
    public:
        CubeResult(Pds::Eb::ResultType rtype) : m_resultType(rtype) {}
    public:
        std::vector<unsigned> add_bins    (const Pds::Eb::ResultDgram&) const;
        std::vector<unsigned> monitor_bins(const Pds::Eb::ResultDgram&) const;
        std::vector<unsigned> record_bins (const Pds::Eb::ResultDgram&) const;
        std::vector<unsigned> flush_bins  (const Pds::Eb::ResultDgram&) const;
        bool                  update_monitor  (const Pds::Eb::ResultDgram&) const;
        bool                  update_record   (const Pds::Eb::ResultDgram&) const;
        bool                  flush           (const Pds::Eb::ResultDgram&) const;
    private:
        Pds::Eb::ResultType m_resultType;
    };
};

#endif
