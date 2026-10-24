/**
 * @file
 * @brief CubeResult, which decodes the cube or window binning fields of a trigger ResultDgram.
 */
#ifndef Pds_CubeResult_hh
#define Pds_CubeResult_hh

#include "psdaq/eb/src/ResultDgram.hh"
#include <vector>

namespace Drp {

    /** Decodes the binning fields of a ResultDgram, reading it as a CubeResultDgram when the result type is Cube and as a WindowResultDgram otherwise. */
    class CubeResult {
    public:
        /** Store the result type that selects the decoding. */
        CubeResult(Pds::Eb::ResultType rtype) : m_resultType(rtype) {}
    public:
        /** For Cube results return the single bin index; otherwise return the indices of the set bits of the window add field, lowest first. */
        std::vector<unsigned> add_bins    (const Pds::Eb::ResultDgram&) const;
        /** For Cube results return the single bin index; otherwise return the indices of the set bits of the window monitor field, lowest first. */
        std::vector<unsigned> monitor_bins(const Pds::Eb::ResultDgram&) const;
        /** For Cube results return the single bin index; otherwise return the indices of the set bits of the window record field, lowest first. */
        std::vector<unsigned> record_bins (const Pds::Eb::ResultDgram&) const;
        /** For Cube results return the single bin index; otherwise return the indices of the set bits of the window flush field, lowest first. */
        std::vector<unsigned> flush_bins  (const Pds::Eb::ResultDgram&) const;
        /** Return the update-monitor bit for Cube results, otherwise whether the window monitor field is non-zero. */
        bool                  update_monitor  (const Pds::Eb::ResultDgram&) const;
        /** Return the update-record bit for Cube results, otherwise whether the window record field is non-zero. */
        bool                  update_record   (const Pds::Eb::ResultDgram&) const;
        /** Return the flush bit for Cube results, otherwise whether the window flush field is non-zero. */
        bool                  flush           (const Pds::Eb::ResultDgram&) const;
    private:
        Pds::Eb::ResultType m_resultType;
    };
};

#endif
