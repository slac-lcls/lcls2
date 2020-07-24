#include <unistd.h>                     // gethostname()
#include <iostream>
#include <fstream>
#include <iomanip>
#include <bitset>
#include <climits>                      // HOST_NAME_MAX
#include "psdaq/service/EbDgram.hh"
#include <DmaDriver.h>
#include "DrpBase.hh"
#include "RunInfoDef.hh"
#include "psalg/utils/SysLog.hh"
#include "xtcdata/xtc/Smd.hh"

#include "rapidjson/document.h"

using namespace XtcData;
using json = nlohmann::json;
using logging = psalg::SysLog;

static void local_mkdir (const char * path);
static json createFileReportMsg(std::string path, std::string absolute_path,
                                timespec create_time, timespec modify_time,
                                unsigned run_num, std::string hostname);
static json createPulseIdMsg(uint64_t pulseId);

namespace Drp {

static long readInfinibandCounter(const std::string& counter)
{
    std::string path{"/sys/class/infiniband/mlx5_0/ports/1/counters/" + counter};
    std::ifstream in(path);
    if (in.is_open()) {
        std::string line;
        std::getline(in, line);
        return stol(line);
    }
    else {
        return 0;
    }
}

static unsigned nextPowerOf2(unsigned n)
{
    unsigned count = 0;

    if (n && !(n & (n - 1))) {
        return n;
    }

    while( n != 0) {
        n >>= 1;
        count += 1;
    }

    return 1 << count;
}

MemPool::MemPool(const Parameters& para) :
    m_transitionBuffers(para.nTrBuffers)
{
    m_fd = open(para.device.c_str(), O_RDWR);
    if (m_fd < 0) {
        logging::critical("Error opening %s", para.device.c_str());
        throw "Error opening kcu1500!!\n";
    }

    uint32_t dmaCount;
    dmaBuffers = dmaMapDma(m_fd, &dmaCount, &m_dmaSize);
    if (dmaBuffers == NULL ) {
        logging::critical("Failed to map dma buffers!");
        throw "Error calling dmaMapDma!!\n";
    }
    logging::info("dmaCount %u  dmaSize %u", dmaCount, m_dmaSize);

    // make sure there are more buffers in the pebble than in the pgp driver
    // otherwise the pebble buffers will be overwritten by the pgp event builder
    m_nbuffers = nextPowerOf2(dmaCount);

    // make the size of the pebble buffer that will contain the datagram equal
    // to the dmaSize times the number of lanes
    // Also include space in the pebble for a pool of transition buffers of
    // worst case size so that they will be part of the memory region that can
    // be RDMAed from to the MEB
    m_bufferSize = __builtin_popcount(para.laneMask) * m_dmaSize;
    pebble.resize(m_nbuffers, m_bufferSize, para.nTrBuffers, para.maxTrSize);
    logging::info("nbuffers %u  pebble buffer size %u", m_nbuffers, m_bufferSize);

    pgpEvents.resize(m_nbuffers);

    // Put the transition buffer pool at the end of the pebble buffers
    uint8_t* buffer = pebble[m_nbuffers];
    for (unsigned i = 0; i < para.nTrBuffers; i++) {
        m_transitionBuffers.push(&buffer[i * para.maxTrSize]);
    }
}

Pds::EbDgram* MemPool::allocateTr()
{
    void* dgram = nullptr;
    if (!m_transitionBuffers.try_pop(dgram)) {
        // See comment for setting para.nTrBuffers in drp.cc
        logging::critical("Empty transition buffer pool");
        throw "Empty transition buffer pool";
    }
    return static_cast<Pds::EbDgram*>(dgram);
}


EbReceiver::EbReceiver(const Parameters& para, Pds::Eb::TebCtrbParams& tPrms,
                       MemPool& pool, ZmqSocket& inprocSend, Pds::Eb::MebContributor& mon,
                       const std::shared_ptr<Pds::MetricExporter>& exporter) :
  EbCtrbInBase(tPrms, exporter),
  m_pool(pool),
  m_mon(mon),
  m_fileWriter(4194304),
  m_smdWriter(4194304),
  m_writing(false),
  m_inprocSend(inprocSend),
  m_count(0),
  m_offset(0),
  m_configureBuffer(para.maxTrSize),
  m_damage(0)
{
    std::map<std::string, std::string> labels
    {{"instrument", para.instrument},
        {"partition", std::to_string(para.partition)},
          {"detname", para.detName}};
    exporter->add("DRP_Damage"    , labels, Pds::MetricType::Gauge  , [&](){ return m_damage; });
    exporter->add("DRP_RecordSize", labels, Pds::MetricType::Counter, [&](){ return m_offset; });
    exporter->add("DRP_RecordDepth", labels, Pds::MetricType::Gauge , [&](){ return m_fileWriter.depth(); });
    m_dmgType = exporter->add("DRP_DamageType", labels, 16);
}

std::string EbReceiver::openFiles(const Parameters& para, const RunInfo& runInfo, std::string hostname, unsigned nodeId)
{
    std::string retVal = std::string{};     // return empty string on success
    if (runInfo.runNumber) {
        std::ostringstream ss;
        ss << runInfo.experimentName <<
              "-r" << std::setfill('0') << std::setw(4) << runInfo.runNumber <<
              "-s" << std::setw(3) << nodeId <<
              "-c000";
        std::string runName = ss.str();
        // data
        std::string dataDir = {para.outputDir + "/" + para.instrument + "/" + runInfo.experimentName + "/xtc"};
        local_mkdir(dataDir.c_str());
        std::string path = {"/" + para.instrument + "/" + runInfo.experimentName + "/xtc/" + runName + ".xtc2"};
        std::string absolute_path = {para.outputDir + path};
        // cpo suggests leaving this print statement in because
        // filesystems can hang in ways we can't timeout/detect
        // and this print statement may speed up debugging significantly.
        std::cout << "Opening file " << absolute_path << std::endl;
        logging::info("Opening file '%s'", absolute_path.c_str());
        if (m_fileWriter.open(absolute_path) == 0) {
            timespec tt; clock_gettime(CLOCK_REALTIME,&tt);
            json msg = createFileReportMsg(path, absolute_path, tt, tt, runInfo.runNumber, hostname);
            m_inprocSend.send(msg.dump());
        } else if (retVal.empty()) {
            retVal = {"Failed to open file '" + absolute_path + "'"};
        }
        // smalldata
        std::string smalldataDir = {para.outputDir + "/" + para.instrument + "/" + runInfo.experimentName + "/xtc/smalldata"};
        local_mkdir(smalldataDir.c_str());
        std::string smalldata_path = {"/" + para.instrument + "/" + runInfo.experimentName + "/xtc/smalldata/" + runName + ".smd.xtc2"};
        std::string smalldata_absolute_path = {para.outputDir + smalldata_path};
        logging::info("Opening file '%s'", smalldata_absolute_path.c_str());
        if (m_smdWriter.open(smalldata_absolute_path) == 0) {
            timespec tt; clock_gettime(CLOCK_REALTIME,&tt);
            json msg = createFileReportMsg(smalldata_path, smalldata_absolute_path, tt, tt, runInfo.runNumber, hostname);
            m_inprocSend.send(msg.dump());
        } else if (retVal.empty()) {
            retVal = {"Failed to open file '" + smalldata_absolute_path + "'"};
        }
        if (retVal.empty()) {
            m_writing = true;
        }
    }
    return retVal;
}

std::string EbReceiver::closeFiles()
{
    logging::debug("%s: m_writing is %s", __PRETTY_FUNCTION__, m_writing ? "true" : "false");
    if (m_writing) {
        m_writing = false;
        logging::debug("calling m_smdWriter.close()...");
        m_smdWriter.close();
        logging::debug("calling m_fileWriter.close()...");
        m_fileWriter.close();
    }
    return std::string{};
}

void EbReceiver::resetCounters()
{
    EbCtrbInBase::resetCounters();

    m_lastIndex = 0;
    m_damage = 0;
    m_dmgType->clear();
}

void EbReceiver::_writeDgram(XtcData::Dgram* dgram)
{
    size_t size = sizeof(*dgram) + dgram->xtc.sizeofPayload();
    m_fileWriter.writeEvent(dgram, size, dgram->time);

    // small data writing
    Smd smd;
    XtcData::NamesId namesId(dgram->xtc.src.value(), NamesIndex::OFFSETINFO);
    XtcData::Dgram* smdDgram = smd.generate(dgram, m_smdWriter.buffer, m_offset, size,
            m_smdWriter.namesLookup, namesId);
    m_smdWriter.writeEvent(smdDgram, sizeof(XtcData::Dgram) + smdDgram->xtc.sizeofPayload(), smdDgram->time);
    m_offset += size;
}

void EbReceiver::process(const Pds::Eb::ResultDgram& result, const void* appPrm)
{
    unsigned index = (uintptr_t)appPrm;
    Pds::EbDgram* dgram = (Pds::EbDgram*)m_pool.pebble[index];
    XtcData::TransitionId::Value transitionId = dgram->service();
    if (transitionId == 0) {
        logging::warning("transitionId == 0 in %s", __PRETTY_FUNCTION__);
    }
    uint64_t pulseId = dgram->pulseId();
    if (pulseId == 0) {
      logging::critical("%spulseId %14lx, ts %u.%09u, tid %d, env %08x%s",
                        RED_ON, pulseId, dgram->time.seconds(), dgram->time.nanoseconds(), dgram->service(), dgram->env, RED_OFF);
    }

    if (index != ((m_lastIndex + 1) & (m_pool.nbuffers() - 1))) {
        logging::critical("%sEbReceiver: jumping index %u  previous index %u  diff %d%s", RED_ON, index, m_lastIndex, index - m_lastIndex, RED_OFF);
        logging::critical("pid     %014lx, tid     %s, env %08x", pulseId, XtcData::TransitionId::name(transitionId), dgram->env);
        logging::critical("lastPid %014lx, lastTid %s", m_lastPid, XtcData::TransitionId::name(m_lastTid));
    }

    if (pulseId != result.pulseId()) {
        logging::critical("pulseIds don't match");
        logging::critical("index %u  previous index %u", index, m_lastIndex);
        uint64_t tPid = pulseId;
        uint64_t rPid = result.pulseId();
        logging::critical("pebble pulseId %014lx, result dgram pulseId %014lx, xor %014lx, diff %ld", tPid, rPid, tPid ^ rPid, tPid - rPid);
        throw "Pulse ID mismatch";
    }

    m_lastIndex = index;
    m_lastPid = pulseId;
    m_lastTid = transitionId;

    // Transfer Result damage to the datagram
    dgram->xtc.damage.increase(result.xtc.damage.value());
    if (dgram->xtc.damage.value()) {
        m_damage++;
        uint16_t damage = dgram->xtc.damage.value();
        while (damage) {
            unsigned dmgType = __builtin_ffsl(damage) - 1;
            damage &= ~(1 << dmgType);
            m_dmgType->observe(dmgType);
        }
    }

    // pass everything except L1 accepts and slow updates to control level
    if ((transitionId != XtcData::TransitionId::L1Accept)) {
        if (transitionId != XtcData::TransitionId::SlowUpdate) {
            if (transitionId == XtcData::TransitionId::Configure) {
                // Cache Configure Dgram for writing out after files are opened
                XtcData::Dgram* configDgram = m_pool.pgpEvents[index].transitionDgram;
                size_t size = sizeof(*configDgram) + configDgram->xtc.sizeofPayload();
                memcpy(m_configureBuffer.data(), configDgram, size);
            }
            if (transitionId == XtcData::TransitionId::BeginRun)
              m_offset = 0;// reset for monitoring (and not recording)
            // send pulseId to inproc so it gets forwarded to the collection
            json msg = createPulseIdMsg(pulseId);
            m_inprocSend.send(msg.dump());
        }

        logging::debug("EbReceiver saw %s transition @ %u.%09u (%014lx)",
                       XtcData::TransitionId::name(transitionId),
                       dgram->time.seconds(), dgram->time.nanoseconds(), pulseId);
    }

    if (m_writing) {                    // Won't ever be true for Configure
        // write event to file if it passes event builder or if it's a transition
        if (result.persist() || result.prescale()) {
            _writeDgram(dgram);
        }
        else if (transitionId != XtcData::TransitionId::L1Accept) {
            if (transitionId == XtcData::TransitionId::BeginRun) {
                m_offset = 0; // reset offset when writing out a new file
                _writeDgram(reinterpret_cast<XtcData::Dgram*>(m_configureBuffer.data()));
            }
            _writeDgram(m_pool.pgpEvents[index].transitionDgram);
            if (transitionId == XtcData::TransitionId::EndRun) {
                logging::debug("%s calling closeFiles()", __PRETTY_FUNCTION__);
                closeFiles();
            }
        }
    }

    if (m_mon.enabled()) {
        // L1Accept
        if (result.isEvent()) {
            if (result.monitor()) {
                m_mon.post(dgram, result.monBufNo());
            }
        }
        // Other Transition
        else {
            m_mon.post(m_pool.pgpEvents[index].transitionDgram);
        }
    }

    // Return buffers and reset event.  Careful with order here!
    // index could be reused as soon as dmaRetIndexes() completes
    PGPEvent* event = &m_pool.pgpEvents[index];
    if (!dgram->isEvent()) {
        m_pool.freeTr(event->transitionDgram);
    }
    for (int i=0; i<4; i++) {
        if (event->mask &  (1 << i)) {
            event->mask ^= (1 << i);    // Zero out mask before dmaRetIndexes()
            m_indices[m_count] = event->buffers[i].index;
            m_count++;
            if (m_count == m_size) {
                dmaRetIndexes(m_pool.fd(), m_count, m_indices);
                // std::cout<<"return dma buffers to driver\n";
                m_count = 0;
            }
        }
    }
}


DrpBase::DrpBase(Parameters& para, ZmqContext& context) :
    pool(para), m_para(para), m_inprocSend(&context, ZMQ_PAIR)
{
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    m_hostname = std::string(hostname);

    m_exposer = Pds::createExposer(para.prometheusDir, m_hostname);
    m_exporter = std::make_shared<Pds::MetricExporter>();

    if (m_exposer) {
        m_exposer->RegisterCollectable(m_exporter);
    }

    std::map<std::string, std::string> labels{{"instrument", para.instrument},
                                              {"partition", std::to_string(para.partition)},
                                              {"detname", para.detName}};
    m_exporter->add("drp_port_rcv_rate", labels, Pds::MetricType::Rate,
                    [](){return 4*readInfinibandCounter("port_rcv_data");});

    m_exporter->add("drp_port_xmit_rate", labels, Pds::MetricType::Rate,
                    [](){return 4*readInfinibandCounter("port_xmit_data");});

    m_tPrms.instrument = para.instrument;
    m_tPrms.partition = para.partition;
    m_tPrms.batching  = m_para.kwargs["batching"] != "yes"; // Default to "no"
    m_tPrms.core[0]   = -1;
    m_tPrms.core[1]   = -1;
    m_tPrms.verbose   = para.verbose;
    m_tebContributor = std::make_unique<Pds::Eb::TebContributor>(m_tPrms, m_exporter);

    m_mPrms.instrument = para.instrument;
    m_mPrms.partition = para.partition;
    m_mPrms.maxEvents = 8;
    m_mPrms.maxEvSize = pool.bufferSize();
    m_mPrms.maxTrSize = para.maxTrSize;
    m_mPrms.verbose   = para.verbose;
    m_mebContributor = std::make_unique<Pds::Eb::MebContributor>(m_mPrms, m_exporter);

    m_ebRecv = std::make_unique<EbReceiver>(m_para, m_tPrms, pool, m_inprocSend, *m_mebContributor, m_exporter);

    m_inprocSend.connect("inproc://drp");
}

void DrpBase::shutdown()
{
    m_tebContributor->shutdown();
    m_mebContributor->shutdown();
    m_ebRecv->shutdown();
}

void DrpBase::reset()
{
    shutdown();
}

json DrpBase::connectionInfo(const std::string& ip)
{
    m_tPrms.ifAddr = ip;
    m_tPrms.port.clear();               // Use an ephemeral port

    // Make a guess at the size of the Result entries
    size_t resSizeGuess = sizeof(Pds::EbDgram) + 2  * sizeof(uint32_t);

    int rc = m_ebRecv->startConnection(m_tPrms.port, resSizeGuess);
    if (rc)  throw "Error starting connection";

    json info = {{"drp_port", m_tPrms.port},
                 {"max_ev_size", m_mPrms.maxEvSize},
                 {"max_tr_size", m_mPrms.maxTrSize}};
    return info;
}

std::string DrpBase::connect(const json& msg, size_t id)
{
    // Save a copy of the json so we can use it to connect to the config database on configure
    m_connectMsg = msg;
    m_collectionId = id;

    parseConnectionParams(msg["body"], id);

    // Make a guess at the size of the Input entries
    size_t inpSizeGuess = sizeof(Pds::EbDgram) + 2 * sizeof(uint32_t);

    int rc = m_tebContributor->connect(inpSizeGuess);
    if (rc) {
        return std::string{"TebContributor connect failed"};
    }
    if (m_mPrms.addrs.size() != 0) {
        void* poolBase = (void*)pool.pebble[0];
        size_t poolSize = pool.pebble.size();
        rc = m_mebContributor->connect(m_mPrms, poolBase, poolSize);
        if (rc) {
            return std::string{"MebContributor connect failed"};
        }
    }
    rc = m_ebRecv->connect();
    if (rc) {
        return std::string{"EbReceiver connect failed"};
    }

    return std::string{};
}

std::string DrpBase::configure(const json& msg)
{
    if (setupTriggerPrimitives(msg["body"])) {
        return std::string("Failed to set up TriggerPrimitive(s)");
    }

    int rc = m_tebContributor->configure();
    if (rc) {
        return std::string{"TebContributor configure failed"};
    }

    if (m_mPrms.addrs.size() != 0) {
        void* poolBase = (void*)pool.pebble[0];
        size_t poolSize = pool.pebble.size();
        rc = m_mebContributor->configure(poolBase, poolSize);
        if (rc) {
            return std::string{"MebContributor configure failed"};
        }
    }

    rc = m_ebRecv->configure();
    if (rc) {
        return std::string{"EbReceiver configure failed"};
    }

    printParams();

    // start eb receiver thread
    m_tebContributor->startup(*m_ebRecv);

    m_tebContributor->resetCounters();
    m_mebContributor->resetCounters();
    m_ebRecv->resetCounters();
    return std::string{};
}

std::string DrpBase::beginrun(const json& phase1Info, RunInfo& runInfo)
{
    std::string msg;
    std::string experiment_name;
    unsigned int run_number = 0;
    if (phase1Info.find("run_info") != phase1Info.end()) {
        if (phase1Info["run_info"].find("experiment_name") != phase1Info["run_info"].end()) {
            experiment_name = phase1Info["run_info"]["experiment_name"];
        }
        if (phase1Info["run_info"].find("run_number") != phase1Info["run_info"].end()) {
            run_number = phase1Info["run_info"]["run_number"];
        }
    }
    runInfo.experimentName = experiment_name;
    runInfo.runNumber = run_number;

    logging::debug("expt=\"%s\" runnum=%u",
                   experiment_name.c_str(), run_number);

    // if run_number is nonzero then we are recording
    if (run_number) {
        logging::debug("Recording to directory '%s'", m_para.outputDir.c_str());
        if (m_para.outputDir.empty()) {
            msg = "Cannot record due to missing output directory";
        } else {
            msg = m_ebRecv->openFiles(m_para, runInfo, m_hostname, m_nodeId);
        }
    }
    m_tebContributor->resetCounters();
    m_mebContributor->resetCounters();
    m_ebRecv->resetCounters();
    return msg;
}

void DrpBase::runInfoSupport(Xtc& xtc, NamesLookup& namesLookup)
{
    XtcData::Alg runInfoAlg("runinfo", 0, 0, 1);
    XtcData::NamesId runInfoNamesId(xtc.src.value(), NamesIndex::RUNINFO);
    XtcData::Names& runInfoNames = *new(xtc) XtcData::Names("runinfo", runInfoAlg,
                                                            "runinfo", "", runInfoNamesId);
    RunInfoDef myRunInfoDef;
    runInfoNames.add(xtc, myRunInfoDef);
    namesLookup[runInfoNamesId] = XtcData::NameIndex(runInfoNames);
}

void DrpBase::runInfoData(Xtc& xtc, NamesLookup& namesLookup, const RunInfo& runInfo)
{
    XtcData::NamesId runInfoNamesId(xtc.src.value(), NamesIndex::RUNINFO);
    XtcData::CreateData runinfo(xtc, namesLookup, runInfoNamesId);
    runinfo.set_string(RunInfoDef::EXPT, runInfo.experimentName.c_str());
    runinfo.set_value(RunInfoDef::RUNNUM, runInfo.runNumber);
}

std::string DrpBase::endrun(const json& phase1Info)
{
    return std::string{};
}

void DrpBase::unconfigure()
{
    m_tebContributor->unconfigure();
    if (m_mPrms.addrs.size() != 0) {
        m_mebContributor->unconfigure();
    }
    m_ebRecv->unconfigure();
}

void DrpBase::disconnect()
{
    m_tebContributor->disconnect();
    if (m_mPrms.addrs.size() != 0) {
        m_mebContributor->disconnect();
    }
    m_ebRecv->disconnect();
}

int DrpBase::setupTriggerPrimitives(const json& body)
{
    using namespace rapidjson;

    Document top;
    const std::string configAlias   = body["config_alias"];
    const std::string triggerConfig = body["trigger_config"];

    // In the following, _0 is added in prints to show the default segment number
    logging::info("Fetching trigger info from ConfigDb/%s/%s_0",
                  configAlias.c_str(), triggerConfig.c_str());

    if (Pds::Trg::fetchDocument(m_connectMsg.dump(), configAlias, triggerConfig, top))
    {
        logging::error("%s:\n  Document '%s_0' not found in ConfigDb",
                       __PRETTY_FUNCTION__, triggerConfig.c_str());
        return -1;
    }

    bool buildAll = top.HasMember("buildAll") && top["buildAll"].GetInt()==1;
    if (!buildAll && !top.HasMember(m_para.detName.c_str())) {
        logging::warning("This DRP is not contributing trigger input data: "
                         "'%s' not found in ConfigDb for %s",
                         m_para.detName.c_str(), triggerConfig.c_str());
        m_tPrms.contractor = 0;    // This DRP won't provide trigger input data
        m_triggerPrimitive = nullptr;
        m_tPrms.maxInputSize = sizeof(Pds::EbDgram); // Revisit: 0 is bad
        return 0;
    }
    m_tPrms.contractor = m_tPrms.readoutGroup;

    std::string symbol("create_producer");
    if (!buildAll)  symbol +=  "_" + m_para.detName;
    m_triggerPrimitive = m_trigPrimFactory.create(top, triggerConfig, symbol);
    if (!m_triggerPrimitive) {
        logging::error("%s:\n  Failed to create TriggerPrimitive",
                       __PRETTY_FUNCTION__);
        return -1;
    }
    m_tPrms.maxInputSize = sizeof(Pds::EbDgram) + m_triggerPrimitive->size();

    if (m_triggerPrimitive->configure(top, m_connectMsg, m_collectionId)) {
        logging::error("%s:\n  Failed to configure TriggerPrimitive",
                       __PRETTY_FUNCTION__);
        return -1;
    }

    return 0;
}

void DrpBase::parseConnectionParams(const json& body, size_t id)
{
    std::string stringId = std::to_string(id);
    logging::debug("id %zu", id);
    m_tPrms.id = body["drp"][stringId]["drp_id"];
    m_mPrms.id = m_tPrms.id;
    m_nodeId = body["drp"][stringId]["drp_id"];

    uint64_t builders = 0;
    m_tPrms.addrs.clear();
    m_tPrms.ports.clear();
    for (auto it : body["teb"].items()) {
        unsigned tebId = it.value()["teb_id"];
        std::string address = it.value()["connect_info"]["nic_ip"];
        std::string port = it.value()["connect_info"]["teb_port"];
        logging::debug("TEB: %u  %s:%s", tebId, address.c_str(), port.c_str());
        builders |= 1ul << tebId;
        m_tPrms.addrs.push_back(address);
        m_tPrms.ports.push_back(port);
    }
    m_tPrms.builders = builders;

    // Store our readout group as a mask to make comparison with Dgram::readoutGroups() cheaper
    m_tPrms.readoutGroup = 1 << unsigned(body["drp"][stringId]["det_info"]["readout"]);
    m_tPrms.contractor = 0;             // Overridden during Configure

    // Build readout group mask for ignoring other partitions' RoGs
    // Also prepare the mask for placing the service bits at the top of Env
    m_para.rogMask = 0x00ff0000;
    for (auto it : body["drp"].items()) {
        unsigned rog = unsigned(it.value()["det_info"]["readout"]);
        if (rog < Pds::Eb::NUM_READOUT_GROUPS - 1) {
            m_para.rogMask |= 1 << rog;
        }
        else {
          logging::error("Ignoring Readout Group %d > max (%d)", rog, Pds::Eb::NUM_READOUT_GROUPS - 1);
        }
    }

    m_mPrms.addrs.clear();
    m_mPrms.ports.clear();
    m_mPrms.maxEvents = 0;
    if (body.find("meb") != body.end()) {
        for (auto it : body["meb"].items()) {
            unsigned mebId = it.value()["meb_id"];
            std::string address = it.value()["connect_info"]["nic_ip"];
            std::string port = it.value()["connect_info"]["meb_port"];
            logging::debug("MEB: %u  %s:%s", mebId, address.c_str(), port.c_str());
            m_mPrms.addrs.push_back(address);
            m_mPrms.ports.push_back(port);
            unsigned count = it.value()["connect_info"]["buf_count"];
            if (!m_mPrms.maxEvents)  m_mPrms.maxEvents = count;
            if (count != m_mPrms.maxEvents) {
                logging::error("maxEvents must be the same for all MEBs");
            }
        }
    }
}

void DrpBase::printParams() const
{
    using namespace Pds::Eb;

    printf("\nParameters of Contributor ID %d (%s:%s):\n",       m_tPrms.id,
                                                                 m_tPrms.ifAddr.c_str(), m_tPrms.port.c_str());
    printf("  Thread core numbers:        %d, %d\n",             m_tPrms.core[0], m_tPrms.core[1]);
    printf("  Partition:                  %d\n",                 m_tPrms.partition);
    printf("  Readout group receipient:   0x%02x\n",             m_tPrms.readoutGroup);
    printf("  Readout group contractor:   0x%02x\n",             m_tPrms.contractor);
    printf("  Bit list of TEBs:           0x%016lx, cnt: %zd\n", m_tPrms.builders,
                                                                 std::bitset<64>(m_tPrms.builders).count());
    printf("  Number of MEBs:             %zd\n",                m_mPrms.addrs.size());
    printf("  Batching state:             %s\n",                 m_tPrms.batching ? "Enabled" : "Disabled");
    printf("  Batch duration:             0x%014lx = %ld uS\n",  BATCH_DURATION, BATCH_DURATION);
    printf("  Batch pool depth:           0x%08x = %u\n",        MAX_BATCHES, MAX_BATCHES);
    printf("  Max # of entries / batch:   0x%08x = %u\n",        MAX_ENTRIES, MAX_ENTRIES);
    printf("  # of TEB contrib. buffers:  0x%08x = %u\n",        MAX_LATENCY, MAX_LATENCY);
    printf("  Max TEB contribution size:  0x%08zx = %zu\n",      m_tPrms.maxInputSize, m_tPrms.maxInputSize);
    printf("  Max MEB L1Accept     size:  0x%08zx = %zu\n",      m_mPrms.maxEvSize, m_mPrms.maxEvSize);
    printf("  Max MEB transition   size:  0x%08zx = %zu\n",      m_mPrms.maxTrSize, m_mPrms.maxTrSize);
    printf("  # of MEB contrib. buffers:  0x%08x = %u\n",        m_mPrms.maxEvents, m_mPrms.maxEvents);
    printf("\n");
}

}

#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>

static void local_mkdir (const char * path)
{
    struct stat64 buf;

    if (path && (stat64(path, &buf) != 0)) {
        if (mkdir(path, 0777)) {
            logging::critical("mkdir %s: %m", path);
        }
    }
}

static json createFileReportMsg(std::string path, std::string absolute_path,
                                timespec create_time, timespec modify_time,
                                unsigned run_num, std::string hostname)
{
    char buf[100];
    json msg, body;

    msg["key"] = "fileReport";
    body["path"] = path;
    body["absolute_path"] = absolute_path;
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&create_time.tv_sec));
    body["create_timestamp"] = buf;
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&modify_time.tv_sec));
    body["modify_timestamp"] = buf;
    body["hostname"] = hostname;
    body["gen"] = 2;                // 2 == LCLS-II
    body["run_num"] = run_num;
    msg["body"] = body;
    return msg;
}

static json createPulseIdMsg(uint64_t pulseId)
{
    json msg, body;
    msg["key"] = "pulseId";
    body["pulseId"] = pulseId;
    msg["body"] = body;
    return msg;
}
