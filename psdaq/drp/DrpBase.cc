#include <unistd.h>                     // gethostname()
#include <iostream>
#include <bitset>
#include <climits>                      // HOST_NAME_MAX
#include "psdaq/service/EbDgram.hh"
#include <DmaDriver.h>
#include "DrpBase.hh"
#include "RunInfoDef.hh"
#include "psalg/utils/SysLog.hh"

#include "rapidjson/document.h"

using namespace XtcData;
using json = nlohmann::json;
using logging = psalg::SysLog;

static void local_mkdir (const char * path);

namespace Drp {

static const unsigned PROM_PORT_BASE  = 9200;
static const unsigned MAX_PROM_PORTS  = 100;

unsigned nextPowerOf2(unsigned n)
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
    transitionBuffer(nullptr),
    m_maxTransitionSize(0)
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
    m_bufferSize = __builtin_popcount(para.laneMask) * m_dmaSize;
    pebble.resize(m_nbuffers, m_bufferSize);
    logging::info("nbuffer %u  pebble buffer size %u", m_nbuffers, m_bufferSize);

    pgpEvents.resize(m_nbuffers);

    transitionBuffer = new uint8_t[para.maxTrSize];
    if (!transitionBuffer) {
        logging::critical("Failed to allocate transition buffer of size %zd", para.maxTrSize);
        throw "Failed to allocate transition buffer";
    }
}

MemPool::~MemPool()
{
    if (transitionBuffer) {
        delete[] static_cast<char*>(transitionBuffer);
        transitionBuffer = nullptr;
        m_maxTransitionSize = 0;
    }
}

EbReceiver::EbReceiver(const Parameters& para, Pds::Eb::TebCtrbParams& tPrms,
                       MemPool& pool, ZmqSocket& inprocSend, Pds::Eb::MebContributor* mon,
                       const std::shared_ptr<MetricExporter>& exporter) :
  EbCtrbInBase(tPrms, exporter),
  m_pool(pool),
  m_mon(mon),
  m_fileWriter(4194304),
  m_smdWriter(1048576),
  m_writing(false),
  m_inprocSend(inprocSend),
  m_count(0),
  m_offset(0),
  m_nodeId(tPrms.id)
{
    m_configureBuffer = new uint8_t[para.maxTrSize];
    if (!m_configureBuffer) {
        logging::critical("Failed to allocate Configure buffer of size %zd", para.maxTrSize);
        throw "Failed to allocate Configure buffer";
    }
}

EbReceiver::~EbReceiver()
{
    if (m_configureBuffer) {
        delete[] static_cast<char*>(m_configureBuffer);
        m_configureBuffer = nullptr;
    }
}

std::string EbReceiver::openFiles(const Parameters& para, const RunInfo& runInfo)
{
    if (runInfo.runNumber) {
        std::string runName = {std::to_string(runInfo.runNumber) + "-" + std::to_string(m_nodeId)};
        std::string fileName = {para.outputDir + "/" + runName + ".xtc2"};
        // cpo suggests leaving this print statement in because
        // filesystems can hang in ways we can't timeout/detect
        // and this print statement may speed up debugging significantly.
        std::cout << "Opening file " << fileName << std::endl;
        logging::info("Opening file '%s'", fileName.c_str());
        m_fileWriter.open(fileName);
        // smalldata
        std::string smalldataDir = {para.outputDir + "/smalldata"};
        local_mkdir(smalldataDir.c_str());
        std::string smalldataFileName = {smalldataDir + "/" + runName + ".smd.xtc2"};
        logging::info("Opening file '%s'", smalldataFileName.c_str());
        m_smdWriter.open(smalldataFileName);
        m_writing = true;
    }
    return std::string{};
}

std::string EbReceiver::closeFiles()
{
    if (m_writing) {
        m_writing = false;
        m_fileWriter.close();
        m_smdWriter.close();
    }
    return std::string{};
}

void EbReceiver::resetCounters()
{
    m_lastIndex = 0;
}

void EbReceiver::_writeDgram(XtcData::Dgram* dgram)
{
    size_t size = sizeof(*dgram) + dgram->xtc.sizeofPayload();
    m_fileWriter.writeEvent(dgram, size);

    // small data writing
    XtcData::Dgram& smdDgram = *(XtcData::Dgram*)m_smdWriter.buffer;
    smdDgram.time = dgram->time;
    smdDgram.env = dgram->env;
    XtcData::TypeId tid(XtcData::TypeId::Parent, 0);
    smdDgram.xtc.contains = tid;
    smdDgram.xtc.damage = 0;
    smdDgram.xtc.extent = sizeof(XtcData::Xtc);

    if (dgram->service() == XtcData::TransitionId::Configure) {
        m_smdWriter.addNames(smdDgram.xtc, m_nodeId);
    }

    XtcData::NamesId namesId(m_nodeId, 0);
    XtcData::CreateData smd(smdDgram.xtc, m_smdWriter.namesLookup, namesId);
    smd.set_value(SmdDef::intOffset, m_offset);
    smd.set_value(SmdDef::intDgramSize, size);
    m_smdWriter.writeEvent(&smdDgram, sizeof(XtcData::Dgram) + smdDgram.xtc.sizeofPayload());

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

    if (index != ((m_lastIndex + 1) & (m_pool.nbuffers() - 1))) {
        logging::critical("%sjumping index %u  previous index %u  diff %d%s", RED_ON, index, m_lastIndex, index - m_lastIndex, RED_OFF);
        logging::critical("pid     %014lx, tid     %s, env %08x", pulseId, XtcData::TransitionId::name(transitionId), dgram->env);
        logging::critical("lastPid %014lx, lastTid %s", m_lastPid, XtcData::TransitionId::name(m_lastTid));
    }

    if (pulseId != result.pulseId()) {
        logging::critical("timestamps don't match");
        logging::critical("index %u  previous index %u", index, m_lastIndex);
        uint64_t tPid = pulseId;
        uint64_t rPid = result.pulseId();
        logging::critical("pebble pulseId %014lx, result dgram pulseId %014lx, xor %014lx, diff %ld", tPid, rPid, tPid ^ rPid, tPid - rPid);
        exit(-1);
    }

    m_lastIndex = index;
    m_lastPid = pulseId;
    m_lastTid = transitionId;

    // pass everything except L1 accepts and slow updates to control level
    if ((transitionId != XtcData::TransitionId::L1Accept)) {
        if (transitionId != XtcData::TransitionId::SlowUpdate) {
            if (transitionId == XtcData::TransitionId::Configure) {
                // Cache Configure Dgram for writing out after files are opened
                XtcData::Dgram* configDgram = m_pool.transitionDgram();
                size_t size = sizeof(*configDgram) + configDgram->xtc.sizeofPayload();
                memcpy(m_configureBuffer, configDgram, size);
            }
            // send pulseId to inproc so it gets forwarded to the collection
            m_inprocSend.send(std::to_string(pulseId));
        }
        logging::debug("EbReceiver saw %s transition @ %d.%09d (%014lx)\n",
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
                _writeDgram(_configureDgram());
            }
            _writeDgram(m_pool.transitionDgram());
        }
    }

    if (m_mon) {
        // L1Accept
        if (result.isEvent()) {
            if (result.monitor()) {
                m_mon->post(dgram, result.monBufNo());
            }
        }
        // Other Transition
        else {
          m_mon->post(m_pool.transitionDgram());
        }
    }

    // return buffers and reset event
    PGPEvent* event = &m_pool.pgpEvents[index];
    for (int i=0; i<4; i++) {
        if (event->mask & (1 << i)) {
            m_indices[m_count] = event->buffers[i].index;
            m_count++;
            if (m_count == m_size) {
                dmaRetIndexes(m_pool.fd(), m_count, m_indices);
                // std::cout<<"return dma buffers to driver\n";
                m_count = 0;
            }
        }
    }
    event->mask = 0;
}


DrpBase::DrpBase(Parameters& para, ZmqContext& context) :
    pool(para), m_para(para), m_inprocSend(&context, ZMQ_PAIR)
{
    m_tPrms.partition = para.partition;
    m_tPrms.core[0]   = 18;
    m_tPrms.core[1]   = 19;
    m_tPrms.verbose   = para.verbose;

    m_mPrms.partition = para.partition;
    m_mPrms.maxEvents = 8;
    m_mPrms.maxEvSize = pool.pebble.bufferSize();
    m_mPrms.maxTrSize = para.maxTrSize;
    m_mPrms.verbose   = para.verbose;

    m_inprocSend.connect("inproc://drp");
}

void DrpBase::shutdown()
{
    m_exporter.reset();

    if (m_tebContributor) {
        m_tebContributor->shutdown();
        m_tebContributor.reset();
    }

    if (m_meb) {
        m_meb->shutdown();
        m_meb.reset();
    }

    m_ebRecv.reset();
}

std::string DrpBase::connect(const json& msg, size_t id)
{
    // Save a copy of the json so we can use it to connect to the config database on configure
    m_connectMsg = msg;
    m_collectionId = id;

    parseConnectionParams(msg["body"], id);

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

    logging::debug("%s: expt=\"%s\" runnum=%u",
                   __PRETTY_FUNCTION__, experiment_name.c_str(), run_number);

    // if run_number is nonzero then we are recording
    if (run_number) {
        logging::debug("Recording to directory '%s'", m_para.outputDir.c_str());
        if (m_para.outputDir.empty()) {
            msg = "Cannot record due to missing output directory";
        } else {
            msg = m_ebRecv->openFiles(m_para, runInfo);
        }
    }
    return msg;
}

void DrpBase::runInfoSupport(Xtc& xtc, NamesLookup& namesLookup)
{
    XtcData::Alg runInfoAlg("runinfo", 0, 0, 1);
    XtcData::NamesId runInfoNamesId(m_nodeId, NamesIndex::RUNINFO);
    XtcData::Names& runInfoNames = *new(xtc) XtcData::Names("runinfo", runInfoAlg,
                                                            "runinfo", "", runInfoNamesId);
    RunInfoDef myRunInfoDef;
    runInfoNames.add(xtc, myRunInfoDef);
    namesLookup[runInfoNamesId] = XtcData::NameIndex(runInfoNames);
}

void DrpBase::runInfoData(Xtc& xtc, NamesLookup& namesLookup, const RunInfo& runInfo)
{
    XtcData::NamesId runInfoNamesId(m_nodeId, NamesIndex::RUNINFO);
    XtcData::CreateData runinfo(xtc, namesLookup, runInfoNamesId);
    runinfo.set_string(RunInfoDef::EXPT, runInfo.experimentName.c_str());
    runinfo.set_value(RunInfoDef::RUNNUM, runInfo.runNumber);
}

std::string DrpBase::endrun(const json& phase1Info)
{
    std::string msg = m_ebRecv->closeFiles();
    return msg;
}

std::string DrpBase::configure(const json& msg)
{
    if (setupTriggerPrimitives(msg["body"])) {
        return std::string("Failed to set up TriggerPrimitive(s)");
    }

    if (m_exposer)  m_exposer.reset();
    unsigned port = 0;
    for (unsigned i = 0; i < MAX_PROM_PORTS; ++i) {
        try {
            port = PROM_PORT_BASE + i;
            m_exposer = std::make_unique<prometheus::Exposer>("0.0.0.0:"+std::to_string(port), "/metrics", 1);
            if (i > 0) {
                if ((i < MAX_PROM_PORTS) && !m_para.prometheusDir.empty()) {
                    char hostname[HOST_NAME_MAX];
                    gethostname(hostname, HOST_NAME_MAX);
                    std::string fileName = m_para.prometheusDir + "/drpmon_" + std::string(hostname) + "_" + std::to_string(i) + ".yaml";
                    FILE* file = fopen(fileName.c_str(), "w");
                    if (file) {
                        fprintf(file, "- targets:\n    - '%s:%d'\n", hostname, port);
                        fclose(file);
                    }
                    else {
                        // %m will be replaced by the string strerror(errno)
                        logging::error("Error creating file %s: %m", fileName.c_str());
                    }
                }
                else {
                    logging::warning("Could not start run-time monitoring server");
                }
            }
            break;
        }
        catch(const std::runtime_error& e) {
            logging::debug("Could not start run-time monitoring server on port %d", port);
            logging::debug("%s", e.what());
        }
    }

    m_exporter = std::make_shared<MetricExporter>();
    if (m_exposer) {
        m_exposer->RegisterCollectable(m_exporter);
    }

    // Create all the eb things and do the connections
    m_tebContributor = std::make_unique<Pds::Eb::TebContributor>(m_tPrms, m_exporter);
    int rc = m_tebContributor->configure(m_tPrms);
    if (rc) {
        return std::string{"TebContributor configure failed"};
    }

    if (m_mPrms.addrs.size() != 0) {
        m_meb = std::make_unique<Pds::Eb::MebContributor>(m_mPrms, m_exporter);
        void* poolBase = (void*)pool.pebble[0];
        size_t poolSize = pool.pebble.size();
        rc = m_meb->configure(m_mPrms, poolBase, poolSize);
        if (rc) {
            return std::string{"MebContributor connect failed"};
        }
    }

    m_ebRecv = std::make_unique<EbReceiver>(m_para, m_tPrms, pool, m_inprocSend, m_meb.get(), m_exporter);
    rc = m_ebRecv->configure(m_tPrms);
    if (rc) {
        return std::string{"EbReceiver configure failed"};
    }

    printParams();

    // start eb receiver thread
    m_tebContributor->startup(*m_ebRecv);

    m_ebRecv->resetCounters();
    return std::string{};
}

int DrpBase::setupTriggerPrimitives(const json& body)
{
    using namespace rapidjson;

    Document top;
    const std::string configAlias = body["config_alias"];
    const std::string dummy("tmoTeb");  // Default trigger library
    std::string&      detName = m_para.trgDetName;
    if (m_para.trgDetName.empty())  m_para.trgDetName = dummy;

    logging::info("Fetching trigger info from ConfigDb/%s/%s\n",
           configAlias.c_str(), detName.c_str());

    if (Pds::Trg::fetchDocument(m_connectMsg.dump(), configAlias, detName, top))
    {
        logging::error("%s:\n  Document '%s' not found in ConfigDb\n",
                __PRETTY_FUNCTION__, detName.c_str());
        return -1;
    }

    if ((detName != dummy) && !top.HasMember(m_para.detName.c_str())) {
        printf("%s:\n  Trigger data not contributed: '%s' not found in ConfigDb for %s\n",
               __PRETTY_FUNCTION__, m_para.detName.c_str(), detName.c_str());
        m_tPrms.contractor = 0;    // This DRP won't provide trigger input data
        m_triggerPrimitive = nullptr;
        return 0;
    }
    m_tPrms.contractor = m_tPrms.readoutGroup;

    std::string symbol("create_producer");
    if (detName != dummy)  symbol +=  "_" + m_para.detName;
    m_triggerPrimitive = m_trigPrimFactory.create(top, detName, symbol);
    if (!m_triggerPrimitive) {
        fprintf(stderr, "%s:\n  Failed to create TriggerPrimitive\n",
                __PRETTY_FUNCTION__);
        return -1;
    }
    m_tPrms.maxInputSize = sizeof(Pds::EbDgram) + m_triggerPrimitive->size();

    if (m_triggerPrimitive->configure(top, m_connectMsg, m_collectionId)) {
        fprintf(stderr, "%s:\n  Failed to configure TriggerPrimitive\n",
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
    m_nodeId = body["drp"][stringId]["drp_id"];
    const unsigned numPorts    = Pds::Eb::MAX_DRPS + Pds::Eb::MAX_TEBS + Pds::Eb::MAX_MEBS + Pds::Eb::MAX_MEBS;
    const unsigned tebPortBase = Pds::Eb::TEB_PORT_BASE + numPorts * m_para.partition;
    const unsigned drpPortBase = Pds::Eb::DRP_PORT_BASE + numPorts * m_para.partition;
    const unsigned mebPortBase = Pds::Eb::MEB_PORT_BASE + numPorts * m_para.partition;

    m_tPrms.port = std::to_string(drpPortBase + m_tPrms.id);
    m_mPrms.id = m_tPrms.id;
    m_tPrms.ifAddr = body["drp"][stringId]["connect_info"]["nic_ip"];

    uint64_t builders = 0;
    m_tPrms.addrs.clear();
    m_tPrms.ports.clear();
    for (auto it : body["teb"].items()) {
        unsigned tebId = it.value()["teb_id"];
        std::string address = it.value()["connect_info"]["nic_ip"];
        logging::debug("TEB: %u  %s", tebId, address.c_str());
        builders |= 1ul << tebId;
        m_tPrms.addrs.push_back(address);
        m_tPrms.ports.push_back(std::string(std::to_string(tebPortBase + tebId)));
    }
    m_tPrms.builders = builders;

    // Store our readout group as a mask to make comparison with Dgram::readoutGroups() cheaper
    m_tPrms.readoutGroup = 1 << unsigned(body["drp"][stringId]["det_info"]["readout"]);
    m_tPrms.contractor = 0;             // Overridden during Configure

    // Build readout group mask for ignoring other partitions' RoGs
    m_para.rogMask = 0x00ff0000 | ((1 << Pds::Eb::NUM_READOUT_GROUPS) - 1);
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
            logging::debug("MEB: %u  %s", mebId, address.c_str());
            m_mPrms.addrs.push_back(address);
            m_mPrms.ports.push_back(std::string(std::to_string(mebPortBase + mebId)));
            unsigned count = it.value()["connect_info"]["buf_count"];
            if (!m_mPrms.maxEvents)  m_mPrms.maxEvents = count;
            if (count != m_mPrms.maxEvents) {
                logging::error("maxEvents must be the same for all MEBs");
            }
        }
    }
}

json DrpBase::connectionInfo()
{
    json info = {{"max_ev_size", m_mPrms.maxEvSize},
                 {"max_tr_size", m_mPrms.maxTrSize}};
    return info;
}

void DrpBase::printParams() const
{
    using namespace Pds::Eb;

    printf("\nParameters of Contributor ID %d:\n",               m_tPrms.id);
    printf("  Thread core numbers:        %d, %d\n",             m_tPrms.core[0], m_tPrms.core[1]);
    printf("  Partition:                  %d\n",                 m_tPrms.partition);
    printf("  Readout group receipient:   0x%02x\n",             m_tPrms.readoutGroup);
    printf("  Readout group contractor:   0x%02x\n",             m_tPrms.contractor);
    printf("  Bit list of TEBs:           0x%016lx, cnt: %zd\n", m_tPrms.builders,
                                                                 std::bitset<64>(m_tPrms.builders).count());
    printf("  Number of MEBs:             %zd\n",                m_mPrms.addrs.size());
    printf("  Batch duration:             0x%014lx = %ld uS\n",  BATCH_DURATION, BATCH_DURATION);
    printf("  Batch pool depth:           %d\n",                 MAX_BATCHES);
    printf("  Max # of entries / batch:   %d\n",                 MAX_ENTRIES);
    printf("  # of TEB contrib. buffers:  %d\n",                 MAX_LATENCY);
    printf("  Max TEB contribution size:  %zd\n",                m_tPrms.maxInputSize);
    printf("  Max MEB L1Accept     size:  %zd\n",                m_mPrms.maxEvSize);
    printf("  Max MEB transition   size:  %zd\n",                m_mPrms.maxTrSize);
    printf("  # of MEB contrib. buffers:  %d\n",                 m_mPrms.maxEvents);
    printf("\n");
}

}

#include <sys/stat.h>
#include <sys/types.h>

static void local_mkdir (const char * path)
{
    struct stat64 buf;

    if (path && (stat64(path, &buf) != 0)) {
        if (mkdir(path, 0777)) {
            logging::critical("mkdir %s: %m", path);
        }
    }
}
