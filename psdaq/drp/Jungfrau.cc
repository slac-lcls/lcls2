#include "Jungfrau.hh"
#include "JungfrauData.hh"

#include "PythonConfigScanner.hh"
#include "psalg/detector/UtilsConfig.hh"
#include "psalg/utils/SysLog.hh"
#include "sls/Detector.h"
#include "xtcdata/xtc/Array.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "xtcdata/xtc/VarDef.hh"

#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <signal.h>
#include <fstream>
#include <string>
#include <iostream>
#include <unordered_map>
#include <set>
#include <limits>
#include <chrono>

using logging = psalg::SysLog;
using json = nlohmann::json;

template<class T>
static std::vector<T> split_string(const std::string& msg,
                                   const std::string& inStr,
                                   const std::string& delimiter,
                                   std::function<T(const std::string&)> func)
{
    std::vector<T> tokens;
    size_t nextPos = 0;
    size_t lastPos = 0;
    std::string subStr;
    std::cout << "Will use the following " << msg << ": ";
    while ((nextPos = inStr.find(delimiter, lastPos)) != std::string::npos) {
        subStr = inStr.substr(lastPos, nextPos-lastPos);
        tokens.push_back(func(subStr));
        lastPos = nextPos + 1;
        std::cout << subStr << ", ";
    }
    subStr = inStr.substr(lastPos);
    tokens.push_back(func(subStr));
    std::cout << subStr << "." << std::endl;
    return tokens;
}

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
    return split_string<unsigned>("segment numbers",
                                  inStr,
                                  delimiter,
                                  [](const std::string& str) {
                                      return static_cast<unsigned>(std::stoul(str));
                                  });
}

/**
 * \brief Split a delimited string of detector hostnames into a vector.
 * Hostnames are passed as a keyword argument of the for
 * `slsHosts=blah1_blah2_blah3_blah4` where the blah1,blah2,blah3,blah4 are
 * the detector control interface hostname/ip  corresponding
 * to the panels coming in on lanes in incrementing order.
 * \param inStr Delimited string of segment numbers, e.g. `blah1_blah2_blah3_blah4`
 * \param delimiter Segment number delimiter. Currently "_".
 * \returns slsHosts A vector of std::string that contains the parsed segment numbers.
 */
static std::vector<std::string> split_slsHosts(const std::string& inStr,
                                               const std::string& delimiter=std::string("_"))
{
    return split_string<std::string>("hostnames",
                                     inStr,
                                     delimiter,
                                     [](const std::string& str) { return str; });
}

static std::chrono::nanoseconds secs_to_ns(double secs)
{
    std::chrono::duration<double> dursecs{secs};
    return std::chrono::duration_cast<std::chrono::nanoseconds>(dursecs);
}

static unsigned calc_shm_id(const std::string& device, uint8_t laneMask)
{
    unsigned shm_id = 0;

    // try find the index of the device name
    size_t pos = device.find("_");
    if (pos != std::string::npos) {
        shm_id = std::stoul(device.substr(pos + 1), nullptr, 16);
    }

    return (shm_id << 8) | laneMask;
}

static const std::unordered_map<std::string, sls::defs::gainMode> slsGainEnumMap
{
    {"DYNAMIC", sls::defs::DYNAMIC},
    {"FORCE_SWITCH_G1", sls::defs::FORCE_SWITCH_G1},
    {"FORCE_SWITCH_G2", sls::defs::FORCE_SWITCH_G2},
    //{"FIX_G0", sls::defs::FIX_G0}, //Not recommended for use.
    {"FIX_G1", sls::defs::FIX_G1},
    {"FIX_G2", sls::defs::FIX_G2},
};

static const std::unordered_map<std::string, sls::defs::speedLevel> slsSpeedLevelMap
{
    {"FULL_SPEED", sls::defs::FULL_SPEED},
    {"HALF_SPEED", sls::defs::HALF_SPEED},
    {"QUARTER_SPEED", sls::defs::QUARTER_SPEED},
};

static const std::unordered_map<std::string, sls::defs::detectorSettings> slsDetSettingsMap
{
    {"normal", sls::defs::GAIN0},
    {"high", sls::defs::HIGHGAIN0},
};

class JungfrauDef : public XtcData::VarDef
{
public:
    enum index
    {
        raw,
        frame_cnt,
        timestamp,
        hotPixelThresh,
        numHotPixels,
        maxHotPixels,
    };

    JungfrauDef()
    {
        NameVec.push_back({"raw", XtcData::Name::UINT16, 3});
        NameVec.push_back({"frame_cnt", XtcData::Name::UINT64});
        NameVec.push_back({"timestamp", XtcData::Name::UINT64});
        NameVec.push_back({"hotPixelThresh", XtcData::Name::UINT16}); // ADU Threshold for hot pixel
        NameVec.push_back({"numHotPixels", XtcData::Name::UINT32}); // Number pixels above threshold
        NameVec.push_back({"maxHotPixels", XtcData::Name::UINT32}); // Maximum number of hot pixels before tripping
    }
} RawDef;

namespace Drp {

static Jungfrau* jungfrau = nullptr;

static void sigHandler(int signal)
{
    if (jungfrau) {
        psignal(signal, "jungfrau cleaning up on signal");
        jungfrau->cleanup();
        jungfrau = nullptr;
    }
}

Jungfrau::Jungfrau(Parameters* para, MemPool* pool) :
    BEBDetector(para, pool)
{
    if (para->kwargs.find("timebase")!=para->kwargs.end() &&
        para->kwargs["timebase"]==std::string("119M")) {
        m_debatch = true;
    }
    for (size_t i = 0; i < PGP_MAX_LANES - 1; ++i) {
        if (para->laneMask & (1 << i)) {
            m_nModules++;
        }
    }
    // Jungfrau always needs to take the m_multiSegment path
    m_multiSegment = true; // From BEBDetector

    if (para->kwargs.find("segNums") != para->kwargs.end()) {
        // BEBDetector will use the underscore delimited string to construct
        // config obj correctly in the XTC
        m_segNoStr = para->kwargs["segNums"];

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

    if (para->kwargs.find("slsHosts") != para->kwargs.end()) {
        std::vector<std::string> slsHosts = split_slsHosts(para->kwargs["slsHosts"]);
        if (slsHosts.size() != m_nModules) {
          logging::critical("Number of detector hostnames doesn't match number of panels: %d "
                            "panels, %d hostnames",
                            m_nModules, slsHosts.size());
          abort();
        }
        m_slsHosts = slsHosts;
    } else {
        logging::critical("Must specify a hostname/ip of the jungfrau control interface for each segment! Check cnf.");
        abort();
    }

    virtChan = 0; // Set correct virtual channel to read from.
    m_para->serNo = std::string(""); // Will fill in during connect

    // try connecting to the jungfrau modules and stop acquisition if needed
    try {
        // shm_id must be unique per machine constructing it from the card num + lane mask should work
        unsigned shm_id = calc_shm_id(para->device, para->laneMask);
        m_slsDet = std::make_unique<sls::Detector>(shm_id);

        try {
            m_slsDet->setHostname(m_slsHosts);
        }
        catch (const sls::RuntimeError &err) {
            auto status = m_slsDet->getDetectorStatus();
            if (status.any(sls::defs::RUNNING) || status.any(sls::defs::WAITING)) {
                m_slsDet->stopDetector();
                m_slsDet->setHostname(m_slsHosts);
            } else {
                // if detector wasn't running or stop fails re-raise to outer handler
                throw;
            }
        }
    }
    catch(const sls::RuntimeError &err) {
        logging::critical("Failed to initialize the Jungfrau control interface: %s", err.what());
        abort();
    }

    // call rogue and fpga initialization after the modules are in a known good state
    // if the modules are acquiring and sending packets during this it causes odd behavior
    _init(para->kwargs["epics_prefix"].c_str());

    // register signal handler to cleanup on exit
    jungfrau = this;

    struct sigaction sa;
    sa.sa_handler = sigHandler;
    sa.sa_flags = SA_RESETHAND;

    sigaction(SIGINT ,&sa,NULL);
    sigaction(SIGABRT,&sa,NULL);
    sigaction(SIGTERM,&sa,NULL);
}

Jungfrau::~Jungfrau()
{
    jungfrau = nullptr;
    cleanup();
}

void Jungfrau::cleanup()
{
    if (m_slsDet) {
        m_slsDet->stopDetector();
        int shm_id = m_slsDet->getShmId();
        m_slsDet.reset();
        sls::freeSharedMemory(shm_id);
    }
}

void Jungfrau::_connectionInfo(PyObject*)
{
    try {
        auto moduleIds = m_slsDet->getModuleId();
        auto boardIds = m_slsDet->getSerialNumber();
        auto firmwareVers = m_slsDet->getFirmwareVersion();
        auto softwareVers = m_slsDet->getDetectorServerVersion();

        // clear out any old serial numbers from previous calls
        m_serNos.clear();

        for (size_t i=0; i < m_nModules; ++i) {
            // construct the LCLS1 style Jungfrau serial id
            std::string serNo = _buildDetId(moduleIds[i],
                                            boardIds[i],
                                            firmwareVers[i],
                                            softwareVers[i],
                                            m_slsHosts[i]);

            m_serNos.push_back(serNo);

            if (i > 0) {
                serNo = "_" + serNo;
            }
            m_para->serNo += serNo;
        }
    }
    catch (const sls::RuntimeError &err) {
        logging::critical("Failed to retrieve Jungfrau module info: %s", err.what());
        abort();
    }
}

unsigned Jungfrau::_configure(XtcData::Xtc& xtc, const void* bufEnd, XtcData::ConfigIter& configo)
{
    XtcData::Alg rawAlg("raw", 0, 2, 0);
    for (size_t i=0; i<m_nModules; ++i) {
        // We have multiple DAQ segments per DRP executable (potentially)
        unsigned detSegment;
        if (m_segNos.empty())
            detSegment = m_para->detSegment;
        else
            detSegment = m_segNos[i];

        XtcData::NamesId rawNamesId(nodeId, EventNamesIndex + i);
        XtcData::Names& rawNames = *new (xtc, bufEnd) XtcData::Names(bufEnd,
                                                                     m_para->detName.c_str(),
                                                                     rawAlg,
                                                                     m_para->detType.c_str(),
                                                                     m_serNos[i].c_str(),
                                                                     rawNamesId,
                                                                     detSegment);
        rawNames.add(xtc, bufEnd, RawDef);
        m_namesLookup[rawNamesId] = XtcData::NameIndex(rawNames);
    }

    // Extract data from the configuration object for Jungfrau SDK configuration
    // BEBDetector should have put one in for each module we have
    // Assume all modules have the same configuration names...
    XtcData::Names& configNames = detector::configNames(configo); //psalg/detector/UtilsConfig.hh

    try {
        // Take all of the modules out of the acquiring state
        m_slsDet->stopDetector();

        // configure the fixed trigger settings
        m_slsDet->setTimingMode(sls::defs::TRIGGER_EXPOSURE);
        m_slsDet->setNumberOfTriggers(std::numeric_limits<int64_t>::max());
        m_slsDet->setNumberOfFrames(1);
        // sync the internal frame counter of all the modules
        m_expectedFrameNum = 1;
        m_slsDet->setNextFrameNumber(m_expectedFrameNum);

        // Loop over the modules we have, extracting the same names for each one
        for (size_t mod=0; mod < m_nModules; ++mod) {
             logging::info("Configuring module %zu - [segment: %u, hostname: %s]",
                           mod, m_segNos[mod], m_slsHosts[mod].c_str());
            sls::Positions pos{static_cast<int>(mod)};
            XtcData::ShapesData& shape = configo.getShape(mod);
            XtcData::DescData desc(shape, configo.namesLookup()[shape.namesId()]);
            // check if the chip is powered on
            if (!m_slsDet->getPowerChip(pos).squash()) {
                logging::info("Powering on chip for module %zu", mod);
                m_slsDet->setPowerChip(true, pos);
            }
            for (size_t i=0; i < configNames.num(); ++i) {
                XtcData::Name& name = configNames.get(i);
                // Extract the configuration data values and do Jungfrau SDK configuration...
                if (strcmp(name.name(), "user.bias_voltage_v") == 0) {
                    uint8_t biasVoltage = desc.get_value<uint8_t>(i);
                    logging::info("Setting module %zu bias voltage to %u", mod, biasVoltage);
                    m_slsDet->setHighVoltage(biasVoltage, pos);
                } else if (strcmp(name.name(), "user.hot_pixel_threshold") == 0) {
                    uint16_t threshold = desc.get_value<uint16_t>(i);
                    logging::info("Pixels above %u will be counted as hot pixels.", threshold);
                    m_hotPixelThreshold = threshold;
                } else if (strcmp(name.name(), "user.max_hot_pixels") == 0) {
                    uint32_t maxHotPixels = desc.get_value<uint32_t>(i);
                    logging::info("Maximum number of hot pixels (across all panels) before trip: %u .",
                                  maxHotPixels);
                    m_maxHotPixels = maxHotPixels;
                } else if (strcmp(name.name(), "user.trigger_delay_s") == 0) {
                    double trigDelay = desc.get_value<double>(i);
                    logging::info("Setting module %zu trigger delay to %f s", mod, trigDelay);
                    m_slsDet->setDelayAfterTrigger(secs_to_ns(trigDelay), pos);
                } else if (strcmp(name.name(), "user.exposure_time_s") == 0) {
                    double exposureTime = desc.get_value<double>(i);
                    logging::info("Setting module %zu exposureTime to %f s", mod, exposureTime);
                    m_slsDet->setExptime(secs_to_ns(exposureTime), pos);
                } else if (strcmp(name.name(), "user.exposure_period") == 0) {
                    double exposurePeriod = desc.get_value<double>(i);
                    logging::info("Setting module %zu exposurePeriod to %f s", mod, exposurePeriod);
                    m_slsDet->setPeriod(secs_to_ns(exposurePeriod), pos);
                } else if (strcmp(name.name(), "user.port") == 0) {
                    uint16_t port = desc.get_value<uint16_t>(i);
                    logging::info("Setting module %zu destination udp port to %u", mod, port);
                    m_slsDet->setDestinationUDPPort(port, pos[0]);
                } else if (strcmp(name.name(), "user.gainMode:gainModeEnum") == 0) {
                    uint32_t gainMode = desc.get_value<uint32_t>(i);
                    auto& gainModeEnums = m_configEnums["gainModeEnum"];
                    if (gainModeEnums.find(gainMode) != gainModeEnums.end()) {
                        auto it = slsGainEnumMap.find(gainModeEnums[gainMode]);
                        if (it != slsGainEnumMap.end()) {
                            logging::info("Setting module %zu gainMode to %s",
                                          mod,
                                          sls::ToString(it->second).c_str());
                            m_slsDet->setGainMode(it->second, pos);
                        } else {
                            logging::error("Enum value %s for module %zu is an invalid parameter for setGainMode()",
                                           gainModeEnums[gainMode].c_str(),
                                           mod);
                            return 1;
                        }
                    } else {
                        logging::error("Invalid gainMode enum value for module %zu: %u", mod, gainMode);
                        return 1;
                    }
                } else if (strcmp(name.name(), "user.speedLevel:speedLevelEnum") == 0) {
                    uint32_t speedLevel = desc.get_value<uint32_t>(i);
                    auto& speedLevelEnums = m_configEnums["speedLevelEnum"];
                    if (speedLevelEnums.find(speedLevel) != speedLevelEnums.end()) {
                        auto it = slsSpeedLevelMap.find(speedLevelEnums[speedLevel]);
                        if (it != slsSpeedLevelMap.end()) {
                            logging::info("Setting module %zu speedLevel to %s",
                                          mod,
                                          sls::ToString(it->second).c_str());
                            m_slsDet->setReadoutSpeed(it->second, pos);
                        } else {
                            logging::error("Enum value %s for module %zu is an invalid parameter for setReadoutSpeed()",
                                           speedLevelEnums[speedLevel].c_str(),
                                           mod);
                            return 1;
                        }
                    } else {
                        logging::error("Invalid speedLevel enum value for module %zu: %u", mod, speedLevel);
                        return 1;
                    }
                } else if (strcmp(name.name(), "user.gain0:gain0Enum") == 0) {
                    uint32_t gain0 = desc.get_value<uint32_t>(i);
                    auto& gain0Enums = m_configEnums["gain0Enum"];
                    if (gain0Enums.find(gain0) != gain0Enums.end()) {
                        auto it = slsDetSettingsMap.find(gain0Enums[gain0]);
                        if (it != slsDetSettingsMap.end()) {
                            logging::info("Setting module %zu gain0 to %s",
                                          mod,
                                          sls::ToString(it->second).c_str());
                            m_slsDet->setSettings(it->second, pos);
                        } else {
                            logging::error("Enum value %s for module %zu is an invalid parameter for setSettings()",
                                           gain0Enums[gain0].c_str(),
                                           mod);
                            return 1;
                        }
                    } else {
                        logging::error("Invalid gain0 enum value for module %zu: %u", mod, gain0);
                        return 1;
                    }
                } else if (strcmp(name.name(), "user.jungfrau_mac") == 0) {
                    std::string jungfrauMac(desc.get_array<char>(i).const_data());
                    logging::info("Setting module %zu jungfrauMac to %s", mod, jungfrauMac.c_str());
                    m_slsDet->setSourceUDPMAC(sls::MacAddr(jungfrauMac), pos);
                } else if (strcmp(name.name(), "user.kcu_mac") == 0) {
                    std::string kcuMac(desc.get_array<char>(i).const_data());
                    logging::info("Setting module %zu kcuMac to %s", mod, kcuMac.c_str());
                    m_slsDet->setDestinationUDPMAC(sls::MacAddr(kcuMac), pos);
                } else if (strcmp(name.name(), "user.jungfrau_ip") == 0) {
                    std::string jungfrauIp(desc.get_array<char>(i).const_data());
                    logging::info("Setting module %zu jungfrauIp to %s", mod, jungfrauIp.c_str());
                    m_slsDet->setSourceUDPIP(sls::HostnameToIp(jungfrauIp.c_str()), pos);
                } else if (strcmp(name.name(), "user.kcu_ip") == 0) {
                    std::string kcuIp(desc.get_array<char>(i).const_data());
                    logging::info("Setting module %zu kcuIp to %s", mod, kcuIp.c_str());
                    m_slsDet->setDestinationUDPIP(sls::HostnameToIp(kcuIp.c_str()), pos);
                } else {
                    const char* start = name.name();
                    const char* enumDelimiter = std::strchr(start, ':');
                    if (enumDelimiter) {
                        std::string enumName(start, enumDelimiter - start);
                        std::string enumType(enumDelimiter + 1);
                        uint32_t enumValue = desc.get_value<uint32_t>(i);
                        m_configEnums[enumType][enumValue] = enumName;
                    }
                }
            }
            m_slsDet->validateUDPConfiguration(pos);
        }

        // Put all of the modules in the acquiring state
        m_slsDet->startDetector();
        // startDetector does not check that aquisition actually started, so check it
        auto status = m_slsDet->getDetectorStatus();
        return !status.contains_only(sls::defs::RUNNING, sls::defs::WAITING);
    }
    catch (const sls::RuntimeError &err) {
        logging::error("Failed to configure Jungfrau modules: %s", err.what());
        return 1;
    }
}

uint32_t Jungfrau::_countNumHotPixels(uint16_t* rawData, uint16_t hotPixelThreshold, uint32_t numPixels) {
    uint32_t numHotPixels {0};
    static const uint16_t gain_bits = 3 << 14;
    static const uint16_t data_bits = (1 << 14) - 1;

    for (size_t i=0; i<numPixels; ++i) {
        uint16_t value = rawData[i];
        if (((value & gain_bits) == gain_bits) &&
            ((data_bits - (value & data_bits)) > hotPixelThreshold)) {
            numHotPixels++;
        }
    }

    return numHotPixels;
}

void Jungfrau::_event(XtcData::Xtc& xtc,
                      const void* bufEnd,
                      std::vector< XtcData::Array<uint8_t> >& subframes)
{
    // Will need to loop over modules to extract data from each subframe
    unsigned rawShape[XtcData::MaxRank] = { 1, JungfrauData::Rows, JungfrauData::Cols };
    for (size_t moduleIdx=0; moduleIdx<m_nModules; ++moduleIdx) {
        XtcData::NamesId rawNamesId(nodeId, EventNamesIndex+moduleIdx);
        XtcData::CreateData cd(xtc, bufEnd, m_namesLookup, rawNamesId);

        // first lane is at index 2 other lanes at 4,5,6, etc...
        unsigned subframeIdx = moduleIdx == 0 ? 2 : moduleIdx + 3;
        std::vector<XtcData::Array<uint8_t>> subframesUdp = _subframes(subframes[subframeIdx].data(),
                                                                       subframes[subframeIdx].num_elem(),
                                                                       JungfrauData::PacketNum);

        uint64_t framenum = 0;
        uint64_t timestamp = 0;
        unsigned segNo = m_segNos[moduleIdx];
        const char* slsHost = m_slsHosts[moduleIdx].c_str();
        XtcData::Data& dataXtc = cd.shapesdata().data();
        XtcData::Array<uint16_t> frame = cd.allocate<uint16_t>(JungfrauDef::raw, rawShape);
        // validate the number of packets
        if (subframesUdp.size() < JungfrauData::PacketNum) {
            logging::error("Missing data: lane-seg-host[%zu-%u-%s] contains %zu packets [%zu]",
                           moduleIdx, segNo, slsHost, subframesUdp.size(), JungfrauData::PacketNum);
            dataXtc.damage.increase(XtcData::Damage::MissingData);
        } else if (subframesUdp.size() > JungfrauData::PacketNum) {
            logging::error("Extra data: lane-seg-host[%zu-%u-%s] contains %zu packets [%zu]",
                           moduleIdx, segNo, slsHost, subframesUdp.size(), JungfrauData::PacketNum);
            dataXtc.damage.increase(XtcData::Damage::Truncated);
        } else {
            unsigned numOutOfOrderPackets = 0;
            uint64_t packetCounter[] = {0xfffffffffffffffful, 0xfffffffffffffffful};
            uint8_t* dataPtr = reinterpret_cast<uint8_t*>(frame.data());
            for (uint32_t udpIdx=0; udpIdx < subframesUdp.size(); udpIdx++) {
                // validate the packet size
                if (subframesUdp[udpIdx].num_elem() != JungfrauData::PacketSize) {
                    logging::error("Corrupted data: lane-seg-host[%zu-%u-%s] packet[%u] unexpected size %lu [%zu]",
                                   moduleIdx, segNo, slsHost, udpIdx, subframesUdp[udpIdx].num_elem(), JungfrauData::PacketSize);
                    dataXtc.damage.increase(XtcData::Damage::Corrupted);
                    break;
                }
                JungfrauData::JungfrauPacket* packet = reinterpret_cast<JungfrauData::JungfrauPacket*>(subframesUdp[udpIdx].data());
                // check framenum is consistent
                if (udpIdx == 0) {
                    framenum = packet->header.framenum;
                    timestamp = packet->header.timestamp;
                } else {
                    if (packet->header.framenum != framenum) {
                        logging::error("Out-of-Order data: lane-seg-host[%zu-%u-%s] packet[%u] unexpected framenum %lu [%lu]",
                                       moduleIdx, segNo, slsHost, udpIdx, packet->header.framenum, framenum);
                        dataXtc.damage.increase(XtcData::Damage::OutOfOrder);
                        break;
                    }
                    // this should not happen so if it does the data in the packet is corrupted...
                    if (packet->header.timestamp != timestamp) {
                        logging::error("Corrupted data: lane-seg-host[%zu-%u-%s] packet[%u] unexpected timestamp %lu [%lu]",
                                       moduleIdx, segNo, slsHost, udpIdx, packet->header.timestamp, timestamp);
                        dataXtc.damage.increase(XtcData::Damage::Corrupted);
                        break;
                    }
                }
                // check that the packets have been properly descrambled
                if (packet->header.packetnum != udpIdx) {
                    if (packet->header.packetnum < JungfrauData::PacketNum) {
                        numOutOfOrderPackets++;
                        logging::debug("Out-of-Order data: lane-seg-host[%zu-%u-%s] framenum[%lu] unexpected packetnum %u [%u]",
                                       moduleIdx, segNo, slsHost, packet->header.framenum, packet->header.packetnum, udpIdx);
                    } else {
                        logging::warning("Corrupted data: lane-seg-host[%zu-%u-%s] framenum[%lu] invalid packetnum %u [%u]",
                                         moduleIdx, segNo, slsHost, packet->header.framenum, packet->header.packetnum, udpIdx);
                        // don't copy the invalid packet payload but not flag damage since this may just be an 'extra' packet
                        continue;
                    }
                }

                size_t offset = packet->header.packetnum * JungfrauData::PayloadSize;
                std::memcpy(dataPtr + offset, &packet->data, JungfrauData::PayloadSize);

                // mark down that we have seen the packet
                uint32_t pidx = packet->header.packetnum / (JungfrauData::PacketNum / 2);
                uint32_t poff = packet->header.packetnum % (JungfrauData::PacketNum / 2);
                packetCounter[pidx] &= ~(1ul<<poff);
            }

            // check if packets came out of the expected order
            if (numOutOfOrderPackets > 0) {
                logging::warning("Out-of-Order data: lane-seg-host[%zu-%u-%s] framenum[%lu] unexpected packet order",
                                 moduleIdx, segNo, slsHost, framenum);
            }

            // check that all the expected packets for the frame were seen
            if ((packetCounter[1] != 0) || (packetCounter[0] != 0)) {
                logging::error("Missing data: lane-seg-host[%zu-%u-%s] framenum[%lu] is missing at least one packet %016lx%016lx",
                               moduleIdx, segNo, slsHost, framenum, packetCounter[1], packetCounter[0]);
                dataXtc.damage.increase(XtcData::Damage::MissingData);
            }
        }

        // check the framenum is the expected value
        if (framenum != m_expectedFrameNum) {
          logging::error("Out-of-Order data: lane-seg-host[%zu-%u-%s] unexpected frame num %lu [%lu] -> diff %lu",
                         moduleIdx, segNo, slsHost,
                         framenum, m_expectedFrameNum,
                         framenum - m_expectedFrameNum);
          dataXtc.damage.increase(XtcData::Damage::OutOfOrder);
        }

        uint32_t numHotPixels = _countNumHotPixels(frame.data(),
                                                   m_hotPixelThreshold,
                                                   JungfrauData::Rows*JungfrauData::Cols);
        cd.set_value(JungfrauDef::frame_cnt, framenum);
        cd.set_value(JungfrauDef::timestamp, timestamp);
        cd.set_value(JungfrauDef::hotPixelThresh, m_hotPixelThreshold);
        cd.set_value(JungfrauDef::numHotPixels, numHotPixels);
        cd.set_value(JungfrauDef::maxHotPixels, m_maxHotPixels);

        // Add the damage from this module to the parent Xtc
        xtc.damage.increase(dataXtc.damage.value());
    }

    m_expectedFrameNum++;
}

std::string Jungfrau::_buildDetId(uint64_t sensor_id,
                                  uint64_t board_id,
                                  uint64_t firmware,
                                  std::string software,
                                  std::string hostname)
{
    std::stringstream id;

    // convert the version string into the form the psana expects e.g. 8.0.2 -> 80000002
    uint64_t software_id = 0;
    size_t start = 0, end = 0;
    int shift = 3;
    while (shift >= 0) {
        end = software.find(".", start);
        software_id |= (std::strtoull(software.substr(start, end).c_str(), NULL, 0) << (shift*16));
        shift--;
        if (end != std::string::npos) {
            start = end+1;
        } else {
            break;
        }
    }

    // most Jungfrau modules have a dummy board_id of 0xff,
    // so lookup the mac addr of the board to use instead
    JungfrauId detid;
    if (m_idLookup.has(hostname)) {
        detid = JungfrauId(m_idLookup[hostname], sensor_id);
    } else {
        detid = JungfrauId(board_id, sensor_id);
    }

    id << std::hex << software_id << "-" << firmware << "-" << detid.full();

    return id.str();
}

unsigned Jungfrau::configureScan(const nlohmann::json& scanKeys, XtcData::Xtc& xtc, const void* bufEnd)
{
    NamesId namesId(nodeId, UpdateNamesIndex);
    return m_configScanner->configure(scanKeys,
                                      xtc,
                                      bufEnd,
                                      namesId,
                                      m_namesLookup,
                                      m_segNos,
                                      m_serNos);
}

unsigned Jungfrau::stepScan(const nlohmann::json& stepInfo, Xtc& xtc, const void* bufEnd)
{
    NamesId namesId(nodeId, UpdateNamesIndex);
    // XTC Updates
    if (unsigned ret = m_configScanner->step(stepInfo, xtc, bufEnd, namesId, m_namesLookup, m_segNos, m_serNos))
        return ret;
    // Actual module configuration
    for (size_t mod = 0; mod < m_nModules; ++mod) {
        sls::Positions pos{ static_cast<int>(mod) };
        if (stepInfo.contains("user.gainMode")) {
            unsigned gainMode = stepInfo["user.gainMode"];
            auto& gainModeEnums = m_configEnums["gainModeEnum"];
            if (gainModeEnums.find(gainMode) != gainModeEnums.end()) {
                if (gainModeEnums.find(gainMode) != gainModeEnums.end()) {
                    auto it = slsGainEnumMap.find(gainModeEnums[gainMode]);
                    if (it != slsGainEnumMap.end()) {
                        logging::info("Setting module %zu gainMode to %s",
                                      mod,
                                      sls::ToString(it->second).c_str());
                        m_slsDet->setGainMode(it->second, pos);
                    } else {
                        logging::error("Enum value %s for module %zu is an invalid parameter for setGainMode()",
                                       gainModeEnums[gainMode].c_str(),
                                       mod);
                        return 1;
                    }
                }
            }
        } else if (stepInfo.contains("user.trigger_delay_s")) {
            double trigDelay = stepInfo["user.trigger_delay_s"];
            logging::info("Setting module %zu trigger delay to %f s", mod, trigDelay);
            m_slsDet->setDelayAfterTrigger(secs_to_ns(trigDelay), pos);
        }
    }
    return 0;
}
} // Drp
