#include "Jungfrau.hh"

#include "xtcdata/xtc/Array.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psalg/utils/SysLog.hh"
#include "psalg/detector/UtilsConfig.hh"

#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <string>
#include <iostream>
//#include <unordered_map>
#include <set>

using logging = psalg::SysLog;
using json = nlohmann::json;

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
  namespace JungfrauData {
  struct Stream {
  public:
      Stream() {}
      void descramble(uint16_t** outBuffer) {
          *outBuffer = scrambledData;
      }
      void printValues() const
      {
          std::cout << "For debugging..." << std::endl;
      }
      uint16_t scrambledData[512*1024]; // Or whatever size...
    };

  } // JungfrauData

Jungfrau::Jungfrau(Parameters* para, MemPool* pool) :
    BEBDetector(para, pool)
{
    _init(para->kwargs["epics_prefix"].c_str());

    if (para->kwargs.find("timebase")!=para->kwargs.end() &&
        para->kwargs["timebase"]==std::string("119M"))
        m_debatch = true;
    for (size_t i = 0; i < PGP_MAX_LANES - 1; ++i) {
        if (para->laneMask & (1 << i)) {
            m_nModules++;
        }
    }
    if (m_nModules > 1)
        m_multiSegment = true; // From BEBDetector

    if (para->kwargs.find("segNums") != para->kwargs.end()) {
        m_segNoStr = para->kwargs["segNums"]; // From BEBDetector
        std::vector<unsigned> segNums = split_segNums(para->kwargs["segNums"]);
        if (segNums.size() != m_nModules) {
            logging::critical("Number of detector segments doesn't match number of panels: %d "
                              "panels, %d segments",
                              m_nModules, segNums.size());
            abort();
        }
        m_segNos = segNums;
    } else {
        if (m_nModules > 1) {
            logging::critical("Must specify segNums manually if using multiple segments! Check cnf.");
            abort();
        }
    }
    virtChan = 0; // Set correct virtual channel to read from.
}

unsigned Jungfrau::_configure(XtcData::Xtc& xtc, const void* bufEnd, XtcData::ConfigIter& configo)
{
    XtcData::Alg rawAlg("raw", 0, 1, 0);
    for (size_t i=0; i<m_nModules; ++i) {
        // Grab serial number first...
        std::string serNo = "";
        // We have multiple DAQ segments per DRP executable (potentially)
        unsigned detSegment;
        if (m_segNos.empty()) {
            detSegment = m_para->detSegment;
        } else {
            detSegment = m_segNos[i];
        }
        XtcData::NamesId rawNamesId(nodeId, EventNamesIndex + i);
        XtcData::Names& rawNames = *new (xtc, bufEnd) XtcData::Names(bufEnd,
                                                                     m_para->detName.c_str(),
                                                                     rawAlg,
                                                                     m_para->detType.c_str(),
                                                                     serNo.c_str(),
                                                                     rawNamesId,
                                                                     detSegment);
        XtcData::VarDef v;
        // 3 Dimensional data for raw image
        v.NameVec.push_back(XtcData::Name("raw", XtcData::Name::UINT16, 3));

        // Add other metadata fields??
        //v.NameVec.push_back(XtcData::Name("frame_cnt", XtcData::Name::UINT16));
        //v.NameVec.push_back(XtcData::Name("timestamp", XtcData::Name::UINT64));
        rawNames.add(xtc, bufEnd, v);
        m_namesLookup[rawNamesId] = XtcData::NameIndex(rawNames);
    }

    // Extract data from the configuration object for Jungfrau SDK configuration
    // Assume all modules have the same configuration names...
    XtcData::Names& configNames = detector::configNames(configo); //psalg/detector/UtilsConfig.hh

    std::set<std::string> configParamNames {
          "user.bias_voltage_v",
          "user.trigger_delay_s",
          "user.exposure_time_s",
          "user.exposure_period",
          "user.gainMode",
          "user.speedLevel"
    };
    // Loop over the modules we have, extracting the same names for each one
    for (size_t i=0; i < m_nModules; ++i) {
        XtcData::ShapesData& shape = configo.getShape(i);
        XtcData::DescData desc(shape, configo.namesLookup()[shape.namesId()]);
        //std::map <std::string, union { uint8_t, uint16_t, double }> configParams;
        for (size_t i=0; i < configNames.num(); ++i) {
            XtcData::Name& name = configNames.get(i);
            // Extract the configuration data values and do Jungfrau SDK configuration...
            if (strcmp(name.name(), "user.bias_voltage_v") == 0) {
                double biasVoltage = desc.get_value<uint8_t>(name.name()); // ?? Double, int??
                std::cout << "Bias Voltage: " << biasVoltage << std::endl;
                // configure...
            } else if (strcmp(name.name(), "user.trigger_delay_s") == 0) {
                double trigDelay = desc.get_value<double>(name.name());
                std::cout << "trigDelay: " << trigDelay << std::endl;
                // configure...
            } else if (strcmp(name.name(), "user.exposure_time_s") == 0) {
                double exposureTime = desc.get_value<double>(name.name());
                std::cout << "exposureTime: " << exposureTime << std::endl;
                // configure...
            } else if (strcmp(name.name(), "user.exposure_period") == 0) {
                double exposurePeriod = desc.get_value<double>(name.name());
                std::cout << "exposurePeriod: " << exposurePeriod << std::endl;
                // configure...
            } else if (strcmp(name.name(), "user.port") == 0) {
                uint16_t port = desc.get_value<uint16_t>(name.name());
            } else if (strcmp(name.name(), "user.gainMode") == 0) {
                // do something with gain...
            } else if (strcmp(name.name(), "user.speedLevel") == 0) {
                // do something with speedLevel...
            }
        }
    }
    return 0;
}

void Jungfrau::_event(XtcData::Xtc& xtc,
                      const void* bufEnd,
                      std::vector< XtcData::Array<uint8_t> >& subframes)
{
    // Will need to loop over modules to extract data from each subframe
    unsigned rawShape[XtcData::MaxRank] = { 1, m_nRows, m_nCols };
    for (size_t moduleIdx=0; moduleIdx<m_nModules; ++moduleIdx) {
        XtcData::NamesId rawNamesId(nodeId, EventNamesIndex+moduleIdx);
        XtcData::DescribedData desc(xtc, bufEnd, m_namesLookup, rawNamesId);
        unsigned subframeIdx = 2; // Calculate where data will be...
        JungfrauData::Stream& udpStream = *new (subframes[subframeIdx].data()) JungfrauData::Stream;
        uint16_t* rawData;
        udpStream.descramble(&rawData);
        unsigned dataSize = m_nElems*2; // Bytes
        memcpy(reinterpret_cast<uint8_t*>(desc.data()), reinterpret_cast<uint8_t*>(rawData), dataSize);
        desc.set_data_length(dataSize);
        desc.set_array_shape(0, rawShape);
    }
}
} // Drp
