#include "Jungfrau.hh"
#include "JungfrauData.hh"

#include "xtcdata/xtc/Array.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psalg/utils/SysLog.hh"
#include "psalg/detector/UtilsConfig.hh"
#include "sls/Detector.h"

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

static const std::unordered_map<std::string, sls::defs::gainMode> slsGainEnumMap
{
    {"DYNAMIC", sls::defs::DYNAMIC},
    {"FORCE_SWITCH_G1", sls::defs::FORCE_SWITCH_G1},
    {"FORCE_SWITCH_G2", sls::defs::FORCE_SWITCH_G2},
    {"FIX_G0", sls::defs::FIX_G0},
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
    _init(para->kwargs["epics_prefix"].c_str());

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

    // try connecting to the jungfrau modules
    try {
        // initialize the slsDetector interface
        // TODO shm_id must be unique per machine constructing it from the card num + lane mask should work
        unsigned shm_id = m_para->detSegment;
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
    XtcData::Alg rawAlg("raw", 0, 1, 0);
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

        // Loop over the modules we have, extracting the same names for each one
        for (size_t mod=0; mod < m_nModules; ++mod) {
            sls::Positions pos{static_cast<int>(mod)};
            XtcData::ShapesData& shape = configo.getShape(mod);
            XtcData::DescData desc(shape, configo.namesLookup()[shape.namesId()]);
            std::unordered_map<std::string, std::unordered_map<uint32_t, std::string>> configEnums;
            //std::map <std::string, union { uint8_t, uint16_t, double }> configParams;
            // check if the chip is powered on
            if (!m_slsDet->getPowerChip(pos).squash()) {
                logging::info("Powering on chip for module %d", mod);
                m_slsDet->setPowerChip(true, pos);
            }
            for (size_t i=0; i < configNames.num(); ++i) {
                XtcData::Name& name = configNames.get(i);
                // Extract the configuration data values and do Jungfrau SDK configuration...
                if (strcmp(name.name(), "user.bias_voltage_v") == 0) {
                    uint8_t biasVoltage = desc.get_value<uint8_t>(i);
                    logging::info("Setting module %zu bias voltage to %u", mod, biasVoltage);
                    m_slsDet->setHighVoltage(biasVoltage, pos);
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
                    auto& gainModeEnums = configEnums["gainModeEnum"];
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
                    auto& speedLevelEnums = configEnums["speedLevelEnum"];
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
                    auto& gain0Enums = configEnums["gain0Enum"];
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
                        configEnums[enumType][enumValue] = enumName;
                    }
                }
            }
            m_slsDet->validateUDPConfiguration();
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

void Jungfrau::_event(XtcData::Xtc& xtc,
                      const void* bufEnd,
                      std::vector< XtcData::Array<uint8_t> >& subframes)
{
    // Will need to loop over modules to extract data from each subframe
    unsigned rawShape[XtcData::MaxRank] = { 1, JungfrauData::Rows, JungfrauData::Cols };
    for (size_t moduleIdx=0; moduleIdx<m_nModules; ++moduleIdx) {
        XtcData::NamesId rawNamesId(nodeId, EventNamesIndex+moduleIdx);
        XtcData::DescribedData desc(xtc, bufEnd, m_namesLookup, rawNamesId);

        unsigned subframeIdx = 2; // Calculate where data will be...
        std::vector<XtcData::Array<uint8_t>> subframesUdp = _subframes(subframes[subframeIdx].data(),
                                                                       subframes[subframeIdx].num_elem(),
                                                                       JungfrauData::PacketNum);

        // validate the number of packets
        if (subframesUdp.size() < JungfrauData::PacketNum) {
            logging::error("Missing data: subframe[%u] contains %zu packets [%zu]",
                           subframeIdx, subframesUdp.size(), JungfrauData::PacketNum);
            xtc.damage.increase(XtcData::Damage::MissingData);
            return;
        } else if (subframesUdp.size() > JungfrauData::PacketNum) {
            logging::error("Extra data: subframe[%u] contains %zu packets [%zu]",
                           subframeIdx, subframesUdp.size(), JungfrauData::PacketNum);
            xtc.damage.increase(XtcData::Damage::Truncated);
            return;
        }

        size_t dataSize = 0;
        uint64_t framenum = 0;
        uint8_t* dataPtr = reinterpret_cast<uint8_t*>(desc.data());
        for (uint32_t udpIdx=0; udpIdx < subframesUdp.size(); udpIdx++) {
            // validate the packet size
            if (subframesUdp[udpIdx].num_elem() != JungfrauData::PacketSize) {
                logging::error("Corrupted data: subframe[%u] packet[%u] unexpected size %lu [%zu]",
                               subframeIdx, udpIdx, subframesUdp[udpIdx].num_elem(), JungfrauData::PacketSize);
                xtc.damage.increase(XtcData::Damage::Corrupted);
                return;
            }
            JungfrauData::JungfrauPacket* packet = reinterpret_cast<JungfrauData::JungfrauPacket*>(subframesUdp[udpIdx].data());
            // check that the packets have been properly descrambled
            if (packet->header.packetnum != udpIdx) {
                logging::error("Out-of-Order data: subframe[%u] framenum[%lu] unexpected packetnum %u [%u]",
                               subframeIdx, framenum, packet->header.packetnum, udpIdx);
                xtc.damage.increase(XtcData::Damage::OutOfOrder);
                return;
            }
            // check framenum is consistent
            if (udpIdx == 0) {
                framenum = packet->header.framenum;
            } else {
                if (packet->header.framenum != framenum) {
                    logging::error("Out-of-Order data: subframe[%u] packet[%u] unexpected framenum %lu [%lu]",
                                   subframeIdx, udpIdx, packet->header.framenum, framenum);
                    xtc.damage.increase(XtcData::Damage::OutOfOrder);
                    return;
                }
            }

            std::memcpy(dataPtr, &packet->data, JungfrauData::PayloadSize);
            dataPtr += JungfrauData::PayloadSize;
            dataSize += JungfrauData::PayloadSize;
        }
        desc.set_data_length(dataSize);
        desc.set_array_shape(0, rawShape);
    }
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

} // Drp
