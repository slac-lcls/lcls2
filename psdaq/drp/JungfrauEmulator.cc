#include "JungfrauEmulator.hh"

#include "drp.hh"
#include "psalg/utils/SysLog.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/VarDef.hh"

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <fstream>
#include <iostream>

using json = nlohmann::json;
using logging = psalg::SysLog;

/**
 * \brief Split a delimited string of detector segment numbers into a vector.
 * Segment numbers may be passed as an optional keyword argument of the form
 * `segNums=0_1_2_3` where the 0,1,2,3 are the detector segment numbers corresponding
 * to the panels coming in on lanes in incrementing order.
 * \param inStr Delimited string of segment numbers, e.g. `0_1_2_3`
 * \param delimiter Segment number delimiter. Currently "_".
 * \returns segNums A vector of unsigned that contains the parsed segment numbers.
 */
static std::vector<unsigned> split_segNums(const std::string& inStr,
                                           const std::string& delimiter=std::string("_"))
{
    std::vector<unsigned> segNums;
    size_t nextPos = 0;
    size_t lastPos = 0;
    std::string subStr;
    std::cout << "Will use the following segment numbers: ";
    while ((nextPos = inStr.find(delimiter, lastPos)) != std::string::npos) {
        subStr = inStr.substr(lastPos, nextPos-lastPos);
        segNums.push_back(static_cast<unsigned>(std::stoul(subStr)));
        lastPos = nextPos + 1;
        std::cout << subStr << ", ";
    }
    subStr = inStr.substr(lastPos);
    segNums.push_back(static_cast<unsigned>(std::stoul(subStr)));
    std::cout << subStr << "." << std::endl;
    return segNums;
}

namespace Drp {

JungfrauEmulator::JungfrauEmulator(Parameters* para, MemPool* pool) :
    XpmDetector(para, pool)
{
    // Setup serial number
    // Each DAQ segment can have multiple panels
    char serNo[16];
    std::string trailingId = std::string("-0000000000-0000000000-0000000000-0000000000-0000000000-0000000000");
    para->serNo = std::string("");
    for (size_t i=0; i<PGP_MAX_LANES-1;++i) {
        if (para->laneMask & (1 << i)) {
            snprintf(serNo,
                     sizeof(serNo),
                     "%010x",
                     static_cast<unsigned>(0xfab1ed0000 + para->detSegment*(PGP_MAX_LANES-1)+i));
            if (!para->serNo.empty())
                para->serNo += std::string("-");
            para->serNo += std::string(serNo) + trailingId;
            m_panelSerNos.push_back(std::string(serNo) + trailingId);
            m_nPanels++;
        }
    }

    if (para->kwargs.find("segNums") != para->kwargs.end()) {
        std::vector<unsigned> segNums = split_segNums(para->kwargs["segNums"]);
        if (segNums.size() != m_nPanels) {
            logging::critical("Number of detector segments doesn't match number of panels: %d panels, %d segments",
                              m_nPanels,
                              segNums.size());
            abort();
        }
        m_segNos = segNums;
    }

    if (para->kwargs.find("imgArray") != para->kwargs.end()) {
        std::ifstream imgArrayFile(para->kwargs["imgArray"].c_str(), std::ios::binary);
        if (!imgArrayFile.good()) {
            logging::critical("Issue opening file %s\n", para->kwargs["imgArray"].c_str());
            abort();
        }
        m_substituteRawData.reserve(m_nElems*m_nPanels);
        imgArrayFile.seekg(0, imgArrayFile.end);
        unsigned lenImgArray = imgArrayFile.tellg();
        imgArrayFile.seekg(0, imgArrayFile.beg);
        std::cout << "Loading " << lenImgArray << " bytes as sub data." << std::endl;
        unsigned panelIdx = 0;
        unsigned panelOffset = 0; // Bytes
        for (size_t i = 0; i < PGP_MAX_LANES - 1; ++i) {
            if (para->laneMask & (i << 1)) {
                // Calculate a panel offset in bytes. Multiply by 2 bytes per element
                //unsigned panelOffset = para->detSegment*(PGP_MAX_LANES-1)+i;
                //panelOffset *= 2*m_nElems;
                panelOffset += 2*m_nElems;
                if (panelOffset + 2*m_nElems > lenImgArray) {
                    // Go back to beginning of the file if too big.
                    // Could do something fancier, but...
                    panelOffset = 0;
                }
                imgArrayFile.seekg(panelOffset);
                m_substituteRawData.resize(m_substituteRawData.size()+m_nElems);
                imgArrayFile.read(reinterpret_cast<char*>(m_substituteRawData.data() + panelIdx*m_nElems),
                                  m_nElems*2);
                panelIdx++;
            }
        }
        imgArrayFile.close();
    }
}

unsigned JungfrauEmulator::configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd)
{
    logging::info("JungfrauEmulator configure");

    if (XpmDetector::configure(config_alias, xtc, bufEnd))
        return 1;

    XtcData::Alg rawAlg("raw", 0, 1, 0);
    for (size_t i=0; i<m_nPanels; ++i) {
        unsigned detSegment;
        if (m_segNos.empty()) {
            unsigned daqSegment = m_para->detSegment;
            detSegment = daqSegment*(PGP_MAX_LANES-1) + i;
        } else {
            detSegment = m_segNos[i];
        }
        std::cout << "Configuring detector segment: " << detSegment << std::endl;
        XtcData::NamesId rawNamesId(nodeId, m_rawNamesIndex+i);
        std::string serNo = m_panelSerNos[i];
        XtcData::Names& rawNames = *new(xtc, bufEnd) XtcData::Names(bufEnd,
                                                                    m_para->detName.c_str(),
                                                                    rawAlg,
                                                                    m_para->detType.c_str(),
                                                                    serNo.c_str(),
                                                                    rawNamesId,
                                                                    detSegment);
        XtcData::VarDef vDef;
        vDef.NameVec.push_back(XtcData::Name("raw", XtcData::Name::UINT16, 3));
        rawNames.add(xtc, bufEnd, vDef);
        m_namesLookup[rawNamesId] = XtcData::NameIndex(rawNames);
    }
    return 0;
}

unsigned JungfrauEmulator::beginrun(XtcData::Xtc& xtc, const void* bufEnd, const json& runInfo)
{
    logging::info("JungfrauEmulator beginrun");
    return 0;
}

void JungfrauEmulator::event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event)
{
    // Jungfrau panel size: 512x1024 pixels
    // 1 fiber/panel with up to 7 panels coming in on one node (1 per lane)
    unsigned panelIdx = 0;
    unsigned rawShape[XtcData::MaxRank] = { 1, m_nRows, m_nCols };
    for (int i=0; i<PGP_MAX_LANES - 1; ++i) {
        if (event->mask & (1 << i)) {
            XtcData::NamesId rawNamesId(nodeId, m_rawNamesIndex+panelIdx);
            XtcData::DescribedData desc(dgram.xtc, bufEnd, m_namesLookup, rawNamesId);

            int dataSize;
            uint8_t* rawData;
            if (m_substituteRawData.empty()) {
                dataSize = event->buffers[i].size - 32;
                uint32_t dmaIndex = event->buffers[i].index;
                rawData = ((uint8_t*)m_pool->dmaBuffers[dmaIndex]) + 32;
            } else {
                dataSize = m_nElems*2; // Should be equivalent to above if properly set up
                rawData = reinterpret_cast<uint8_t*>(m_substituteRawData.data() + panelIdx*m_nElems);
            }
            memcpy((uint8_t*)desc.data(), rawData, dataSize);
            desc.set_data_length(dataSize);
            // Create a new desc for each segment, so always set array shape
            // for index 0!
            desc.set_array_shape(0, rawShape);
            panelIdx++;
      }
    }
}
} // Drp
