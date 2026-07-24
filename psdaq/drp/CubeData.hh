#pragma once

#include "xtcdata/xtc/NamesId.hh"
#include "xtcdata/xtc/TransitionId.hh"
#include "xtcdata/xtc/VarDef.hh"

namespace XtcData { 
    class Dgram;
    class ShapesData; 
    class Src;
}

typedef std::vector<XtcData::ShapesData*> SDV;

namespace Pds { class EbDgram; }

namespace Drp {

    class Detector;

    class CubeData {
    public:
        CubeData(Detector& det, unsigned nbins, unsigned poolBufferSize);
        ~CubeData();
    public:
        //  Initialize the cube from the first event
        void initialize(XtcData::Src& src,
                        const SDV&    rawData,
                        int           threadNum=1);   // debugging only
        //  Zero some bins
        void flush     (const std::vector<unsigned>& bins);
        //  Add names to the XTC Configure
        void addNames(unsigned detSegment,
                      XtcData::Dgram* dgram, 
                      const void* bufEnd);
        //  Add to the cube
        void add    (unsigned   bin,
                     const SDV& rawData);
        void addSub (unsigned   bin,
                     unsigned   subDet,
                     const SDV& rawData);
        //  Copy one or more bins into a datagram
        Pds::EbDgram* copyBins(const std::vector<unsigned>& bins,
                               SDV&                         rawDataV,
                               Pds::EbDgram*                dg);
        //  Add the contents of one or more bins into a datagram
        void addBins(const std::vector<unsigned>& bins,
                     const SDV&                   rawDataV,
                     Pds::EbDgram*                dg);
        //  Get the cube as a datagram for EndRun
        std::vector<XtcData::Dgram*> dgram(XtcData::Dgram*, XtcData::TransitionId::Value);
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

