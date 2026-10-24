/**
 * @file
 * @brief CubeData, which accumulates detector raw data into bins (a cube) held in one or more datagram buffers.
 */
#pragma once

#include "xtcdata/xtc/NamesId.hh"
#include "xtcdata/xtc/TransitionId.hh"
#include "xtcdata/xtc/VarDef.hh"

namespace XtcData { 
    class Dgram;
    class ShapesData; 
    class Src;
}

/** Alias for a vector of XtcData::ShapesData pointers (one per raw definition of the detector). */
typedef std::vector<XtcData::ShapesData*> SDV;

namespace Pds { class EbDgram; }

namespace Drp {

    class Detector;

    /** Accumulates the detector raw data (Detector::rawDef()) into nbins bins. The cube is stored as one or more datagrams (each below 2 GiB) holding, per raw definition, the bin indices, entry counts and double-valued sums. */
    class CubeData {
    public:
        /** Take the raw definitions and names indices from det, split the nbins bins over enough buffers to keep each below 2 GiB, allocate those buffers, and allocate a separate buffer (sized for one bin plus poolBufferSize) used by copyBins(). */
        CubeData(Detector& det, unsigned nbins, unsigned poolBufferSize);
        /** Free the cube buffers and the bin buffer. */
        ~CubeData();
    public:
        //  Initialize the cube from the first event
        /** Lay out each cube buffer as a datagram with source src: per raw definition, bin indices, zeroed entry counts and zeroed sums, with array shapes taken from Detector::shapeCube() for rawData. Aborts if array data is needed but rawData is missing, or if a buffer would overflow. threadNum only controls logging (the comment says debugging only). */
        void initialize(XtcData::Src& src,
                        const SDV&    rawData,
                        int           threadNum=1);   // debugging only
        //  Zero some bins
        /** Zero the entry count and all sums of each bin in bins. */
        void flush     (const std::vector<unsigned>& bins);
        //  Add names to the XTC Configure
        /** Add cube Names (algorithm cube 2.0.0, with the detector name, type and ID of the raw names) for each raw definition to dgram and register them in the detector's names lookup. */
        void addNames(unsigned detSegment,
                      XtcData::Dgram* dgram, 
                      const void* bufEnd);
        //  Add to the cube
        /** Add one event to bin: increment its entry count and call Detector::addToCube() (subIndex 0) for every raw value; raw definitions with no data in rawData are skipped. */
        void add    (unsigned   bin,
                     const SDV& rawData);
        /** Like add() for sub-detector subDet (passed to Detector::addToCube() as subIndex), without incrementing the entry count. */
        void addSub (unsigned   bin,
                     unsigned   subDet,
                     const SDV& rawData);
        //  Copy one or more bins into a datagram
        /** Copy dg into the bin buffer and append, per raw definition, a cube entry holding the given bins (indices, entry counts and sums) copied from the cube; the new ShapesData pointers are appended to rawDataV. Returns the copy; aborts if it would overflow the bin buffer. */
        Pds::EbDgram* copyBins(const std::vector<unsigned>& bins,
                               SDV&                         rawDataV,
                               Pds::EbDgram*                dg);
        //  Add the contents of one or more bins into a datagram
        /** Add the entry counts and sums of the given bins of the cube into the entries described by rawDataV (as produced by copyBins()); dg is not used. */
        void addBins(const std::vector<unsigned>& bins,
                     const SDV&                   rawDataV,
                     Pds::EbDgram*                dg);
        //  Get the cube as a datagram for EndRun
        /** Overwrite the transition header of every cube buffer with the type, time and env of dgram and transition ID transitionId (bit 16 of env is set on all but the last buffer) and return the buffers as datagrams. */
        std::vector<XtcData::Dgram*> dgram(XtcData::Dgram*, XtcData::TransitionId::Value);
        /** Add the entry counts and sums of every bin of this cube into the matching cube datagrams in the argument (one per buffer). */
        void                         add  (std::vector<XtcData::Dgram*>&);

    private:
        Detector&                     m_det;
        unsigned                      m_nbins;
        unsigned                      m_binsPerBuf;
        unsigned                      m_bufferSize;
        unsigned                      m_bufferBinSize;
        std::vector<XtcData::VarDef>  m_rawDefV;
        std::vector<XtcData::VarDef>  m_cubeDefV;
        std::vector<XtcData::NamesId> m_rawNames;
        std::vector<char*>            m_buffer;
        char*                         m_bufferBin;
        bool                          m_init;
    };
}

