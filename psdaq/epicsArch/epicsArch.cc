#include "EpicsArch.hh"

#include <cassert>
#include <bitset>
#include <chrono>
#include <unistd.h>
#include <iostream>
#include <map>
#include <algorithm>
#include <getopt.h>
#include <sys/prctl.h>
#include <Python.h>
#include "psdaq/aes-stream-drivers/DataDriver.h"
#include "drp/TebReceiver.hh"
#include "drp/RunInfoDef.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/fast_monotonic_clock.hh"

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

using namespace XtcData;
using namespace Pds;
using json = nlohmann::json;
using logging = psalg::SysLog;
using ms_t = std::chrono::milliseconds;

namespace Drp {

Pgp::Pgp(const Parameters& para, MemPool& pool, Detector* det) :
    PgpReader(para, pool, MAX_RET_CNT_C, 32),
    m_det(det),
    m_available(0), m_current(0), m_nDmaRet(0)
{
    if (pool.setMaskBytes(para.laneMask, det->virtChan)) {
        logging::error("Failed to allocate lane/vc");
    }
}

EbDgram* Pgp::_handle(uint32_t& evtIndex)
{
    const TimingHeader* timingHeader = handle(m_det, m_current);
    if (!timingHeader)  return nullptr;

    uint32_t pgpIndex = timingHeader->evtCounter & (m_pool.nDmaBuffers() - 1);
    PGPEvent* event = &m_pool.pgpEvents[pgpIndex];

    // make new dgram in the pebble
    // It must be an EbDgram in order to be able to send it to the MEB
    evtIndex = event->pebbleIndex;
    Src src = m_det->nodeId;
    EbDgram* dgram = new(m_pool.pebble[evtIndex]) EbDgram(*timingHeader, src, m_para.rogMask);

    // Collect indices of DMA buffers that can be recycled and reset event
    freeDma(event);

    return dgram;
}

EbDgram* Pgp::next(uint32_t& evtIndex)
{
    // get new buffers
    if (m_current == m_available) {
        m_current = 0;
        m_available = read();
        m_nDmaRet = m_available;
        if (m_available == 0) {
            return nullptr;
        }
    }

    EbDgram* dgram = _handle(evtIndex);
    m_current++;
    return dgram;
}

// ---

EaDetector::EaDetector(Parameters& para, const std::string& pvCfgFile, MemPoolCpu& pool) :
    XpmDetector(&para, &pool),
    m_pvCfgFile(pvCfgFile)
{
    virtChan = 0;
}

EaDetector::~EaDetector()
{
    EpicsArchMonitor::close();
}

unsigned EaDetector::connect(const json& connectJson, const std::string& collectionId, std::string& msg)
{
    XpmDetector::connect(connectJson, collectionId);

    try
    {
        m_monitor = std::make_unique<EpicsArchMonitor>(m_pvCfgFile.c_str(), m_para->verbose, msg);
    }
    catch(std::string& error)
    {
        logging::error("Failed to create EpicsArchMonitor( %s ): %s",
                       m_pvCfgFile.c_str(), error.c_str());
        m_monitor.reset();
        msg = error;
        return 1;
    }

    unsigned pvCount = 0;
    unsigned tmo = 1;                   // Seconds
    unsigned nNotConnected = m_monitor->validate(pvCount, tmo);
    if (nNotConnected) {
        msg = "Number of PVs that didn't connect: " + std::to_string(nNotConnected) + " (of " + std::to_string(pvCount) + ")";
        if (nNotConnected == pvCount) {
            return 1;
        }
    }

    return 0;
}

unsigned EaDetector::disconnect()
{
    XpmDetector::shutdown();

    m_monitor.reset();
    return 0;
}

unsigned EaDetector::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
    logging::info("EpicsArch configure");

    if (XpmDetector::configure(config_alias, xtc, bufEnd))
        return 1;

    m_nStales = 0;

    size_t payloadSize;
    m_monitor->addNames(m_para->detName, m_para->detType, m_para->serNo,
                        m_para->detSegment,
                        xtc, bufEnd, m_namesLookup, nodeId, payloadSize);
    // make sure config transition will fit in the transition buffer (where SlowUpdate goes), however this check is too late: code can unfortunately segfault in the above line -cpo
    if (sizeof(Dgram)+xtc.sizeofPayload() > m_para->maxTrSize) {
        logging::critical("Increase Parameter::maxTrSize (%zd) to avoid truncation of configure transition (%zd)",
                          m_para->maxTrSize, sizeof(Dgram)+xtc.sizeofPayload());
        abort();
    }
    // make sure the data will fit in the transition buffer (where SlowUpdate goes)
    if (payloadSize > m_para->maxTrSize) {
        logging::critical("Increase Parameter::maxTrSize (%zd) to avoid truncation of SlowUpdate data (%zd)",
                         m_para->maxTrSize, payloadSize);
        abort();
    }

    return 0;
}

unsigned EaDetector::unconfigure()
{
    m_namesLookup.clear();   // erase all elements

    return 0;
}

void EaDetector::event(Dgram&, const void* bufEnd, PGPEvent*, uint64_t)
{
    // Unused
}

void EaDetector::slowupdate(Xtc& xtc, const void* bufEnd)
{
    m_monitor->getData(xtc, bufEnd, m_namesLookup, nodeId, m_nStales);
}

// ---

EaDrp::EaDrp(Parameters& para, MemPoolCpu& pool, Detector& det, ZmqContext& context) :
    DrpBase    (para, pool, det, context),
    m_para     (para),
    m_det      (det),
    m_pgp      (para, pool, &det),
    m_terminate(false)
{
    // Set the TebReceiver we will use in the base class
    setTebReceiver(std::make_unique<TebReceiver>(m_para, *this));
}

std::string EaDrp::configure(const json& msg)
{
    std::string errorMsg = DrpBase::configure(msg);
    if (!errorMsg.empty()) {
        return errorMsg;
    }

    m_workerThread = std::thread{&EaDrp::_worker, this};

    return std::string();
}

unsigned EaDrp::unconfigure()
{
    DrpBase::unconfigure(); // TebContributor must be shut down before the worker

    m_terminate.store(true, std::memory_order_release);
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }

    return 0;
}

int EaDrp::_setupMetrics(const std::shared_ptr<MetricExporter> exporter)
{
    std::map<std::string, std::string> labels{{"instrument", m_para.instrument},
                                              {"partition", std::to_string(m_para.partition)},
                                              {"detname", m_para.detName},
                                              {"detseg", std::to_string(m_para.detSegment)},
                                              {"alias", m_para.alias}};
    m_nEvents = 0;
    exporter->add("drp_event_rate", labels, MetricType::Rate,
                  [&](){return m_nEvents;});
    m_nUpdates = 0;
    exporter->add("drp_update_count", labels, MetricType::Counter,
                  [&](){return m_nUpdates;});
    exporter->add("drp_stale_count", labels, MetricType::Counter,
                  [&](){return static_cast<EaDetector&>(m_det).nStales();});

    exporter->add("drp_num_dma_ret", labels, MetricType::Gauge,
                  [&](){return m_pgp.nDmaRet();});
    exporter->add("drp_pgp_byte_rate", labels, MetricType::Rate,
                  [&](){return m_pgp.dmaBytes();});
    exporter->add("drp_dma_size", labels, MetricType::Gauge,
                  [&](){return m_pgp.dmaSize();});
    exporter->add("drp_th_latency", labels, MetricType::Gauge,
                  [&](){return m_pgp.latency();});
    exporter->add("drp_num_dma_errors", labels, MetricType::Gauge,
                  [&](){return m_pgp.nDmaErrors();});
    exporter->add("drp_num_no_common_rog", labels, MetricType::Gauge,
                  [&](){return m_pgp.nNoComRoG();});
    exporter->add("drp_num_missing_rogs", labels, MetricType::Gauge,
                  [&](){return m_pgp.nMissingRoGs();});
    exporter->add("drp_num_th_error", labels, MetricType::Gauge,
                  [&](){return m_pgp.nTmgHdrError();});
    exporter->add("drp_num_pgp_jump", labels, MetricType::Gauge,
                  [&](){return m_pgp.nPgpJumps();});
    exporter->add("drp_num_no_tr_dgram", labels, MetricType::Gauge,
                  [&](){return m_pgp.nNoTrDgrams();});

    exporter->add("drp_num_pgp_in_user", labels, MetricType::Gauge,
                  [&](){return m_pgp.nPgpInUser();});
    exporter->add("drp_num_pgp_in_hw", labels, MetricType::Gauge,
                  [&](){return m_pgp.nPgpInHw();});
    exporter->add("drp_num_pgp_in_prehw", labels, MetricType::Gauge,
                  [&](){return m_pgp.nPgpInPreHw();});
    exporter->add("drp_num_pgp_in_rx", labels, MetricType::Gauge,
                  [&](){return m_pgp.nPgpInRx();});

    return 0;
}

void EaDrp::_worker()
{
    logging::info("EpicsArch worker is starting with process ID %lu", syscall(SYS_gettid));
    if (prctl(PR_SET_NAME, "epicsArch/Worker", 0, 0, 0) == -1) {
        perror("prctl");
    }

    m_terminate.store(false, std::memory_order_release);

    // Reset counters to avoid 'jumping' errors reconfigures
    pool.resetCounters();
    m_pgp.resetEventCounter();

    // Set up monitoring
    auto exporter = std::make_shared<MetricExporter>();
    if (exposer()) {
        exposer()->RegisterCollectable(exporter);

        if (_setupMetrics(exporter))  return;
    }

    while (true) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }

        uint32_t index;
        EbDgram* dgram = m_pgp.next(index);
        if (dgram) {
            m_nEvents++;

            TransitionId::Value service = dgram->service();
            logging::debug("EAWorker saw %s @ %d.%09d (%014lx)",
                           TransitionId::name(service),
                           dgram->time.seconds(), dgram->time.nanoseconds(), dgram->pulseId());
            if (service != TransitionId::L1Accept) {
                // Find the transition dgram in the pool and initialize its header
                EbDgram* trDgram = pool.transitionDgrams[index];
                const void*   bufEnd  = (char*)trDgram + m_para.maxTrSize;
                if (!trDgram)  continue; // Can happen during shutdown
                *trDgram = *dgram;

                if (service == TransitionId::SlowUpdate) {
                    m_nUpdates++;

                    m_det.slowupdate(trDgram->xtc, bufEnd);
                }
                else {
                    // copy the temporary xtc created on phase 1 of the transition
                    // into the real location
                    Xtc& trXtc = m_det.transitionXtc();
                    trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
                    auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
                    memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());
                }
            }

            _sendToTeb(*dgram, index);
        }
        else {
            // Time out batches for the TEB
            tebContributor().timeout();
        }
    }

    // Flush the DMA buffers
    m_pgp.flush();

    if (exposer())  exporter.reset();

    logging::info("Worker thread finished");
}

void EaDrp::_sendToTeb(const EbDgram& dgram, uint32_t index)
{
    // Make sure the datagram didn't get too big
    const size_t size = sizeof(dgram) + dgram.xtc.sizeofPayload();
    const size_t maxSize = (dgram.service() == TransitionId::L1Accept)
                         ? pool.pebble.bufferSize()
                         : m_para.maxTrSize;
    if (size > maxSize) {
        logging::critical("%s Dgram of size %zd overflowed buffer of size %zd", TransitionId::name(dgram.service()), size, maxSize);
        throw "Dgram overflowed buffer";
    }

    auto l3InpBuf = tebContributor().fetch(index);
    EbDgram* l3InpDg = new(l3InpBuf) EbDgram(dgram);
    if (l3InpDg->isEvent()) {
        auto trgPrimitive = triggerPrimitive();
        if (trgPrimitive) { // else this DRP doesn't provide input
            const void* bufEnd = (char*)l3InpDg + sizeof(*l3InpDg) + trgPrimitive->size();
            trgPrimitive->event(pool, index, dgram.xtc, l3InpDg->xtc, bufEnd); // Produce
        }
    }
    tebContributor().process(l3InpDg);
}

// ---

EpicsArchApp::EpicsArchApp(Parameters& para, const std::string& pvCfgFile) :
    CollectionApp(para.collectionHost, para.partition, "drp", para.alias),
    m_para       (para),
    m_pool       (para),
    m_unconfigure(false)
{
    Py_Initialize();                    // for use by configuration

    m_det = std::make_unique<EaDetector>(m_para, pvCfgFile, m_pool);
    m_drp = std::make_unique<EaDrp>(para, m_pool, *m_det, context());

    logging::info("Ready for transitions");
}

EpicsArchApp::~EpicsArchApp()
{
    // Try to take things down gracefully when an exception takes us off the
    // normal path so that the most chance is given for prints to show up
    handleReset(json({}));

    Py_Finalize();                      // for use by configuration
}

void EpicsArchApp::_disconnect()
{
    m_drp->disconnect();
    m_det->disconnect();
}

void EpicsArchApp::_unconfigure()
{
    m_drp->pool.shutdown();             // Release Tr buffer pool
    m_drp->unconfigure();
    m_det->unconfigure();
    m_unconfigure = false;
}

json EpicsArchApp::connectionInfo(const json& msg)
{
    std::string ip = m_para.kwargs.find("ep_domain") != m_para.kwargs.end()
                   ? getNicIp(m_para.kwargs["ep_domain"])
                   : getNicIp(m_para.kwargs["forceEnet"] == "yes");
    logging::debug("nic ip  %s", ip.c_str());
    json body = {{"connect_info", {{"nic_ip", ip}}}};
    json info = static_cast<Detector&>(*m_det).connectionInfo(msg);
    body["connect_info"].update(info);
    json bufInfo = m_drp->connectionInfo(ip);
    body["connect_info"].update(bufInfo);
    return body;
}

void EpicsArchApp::connectionShutdown()
{
    static_cast<Detector&>(*m_det).connectionShutdown();
    m_drp->shutdown();
}

void EpicsArchApp::_error(const std::string& which, const json& msg, const std::string& errorMsg)
{
    json body = json({});
    body["err_info"] = errorMsg;
    json answer = createMsg(which, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void EpicsArchApp::handleConnect(const nlohmann::json& msg)
{
    std::string errorMsg = m_drp->connect(msg, getId());
    if (!errorMsg.empty()) {
        logging::error(("DrpBase::connect: " + errorMsg).c_str());
        _error("connect", msg, errorMsg);
        return;
    }

    m_det->nodeId = m_drp->nodeId();
    unsigned rc = m_det->connect(msg, std::to_string(getId()), errorMsg);
    if (!errorMsg.empty()) {
        if (!rc) {
            logging::warning(("EaDetector::connect: " + errorMsg).c_str());
            json warning = createAsyncWarnMsg(m_para.alias, errorMsg);
            reply(warning);
        }
        else {
            logging::error(("EaDetector::connect: " + errorMsg).c_str());
            _error("connect", msg, errorMsg);
            return;
        }
    }

    json body = json({});
    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void EpicsArchApp::handleDisconnect(const json& msg)
{
    // Carry out the queued Unconfigure, if there was one
    if (m_unconfigure) {
        _unconfigure();
    }

    _disconnect();

    json body = json({});
    reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void EpicsArchApp::handlePhase1(const json& msg)
{
    std::string key = msg["header"]["key"];
    logging::debug("handlePhase1 for %s in EpicsArchApp", key.c_str());

    Xtc& xtc = m_det->transitionXtc();
    xtc = {{TypeId::Parent, 0}, {m_det->nodeId}};
    auto bufEnd = m_det->trXtcBufEnd();

    json phase1Info{ "" };
    if (msg.find("body") != msg.end()) {
        if (msg["body"].find("phase1Info") != msg["body"].end()) {
            phase1Info = msg["body"]["phase1Info"];
        }
    }

    json body = json({});

    if (key == "configure") {
        if (m_unconfigure) {
            _unconfigure();
        }

        // Configure the detector first
        std::string config_alias = msg["body"]["config_alias"];
        unsigned error = m_det->configure(config_alias, xtc, bufEnd);
        if (error) {
            std::string errorMsg = "Failed transition phase 1";
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }

        // Next, configure the DRP
        std::string errorMsg = m_drp->configure(msg);
        if (!errorMsg.empty()) {
            errorMsg = "Phase 1 error: " + errorMsg;
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }

        m_drp->runInfoSupport(xtc, bufEnd, m_det->namesLookup());
        m_drp->chunkInfoSupport(xtc, bufEnd, m_det->namesLookup());
    }
    else if (key == "unconfigure") {
        // "Queue" unconfiguration until after phase 2 has completed
        m_unconfigure = true;
    }
    else if (key == "beginrun") {
        RunInfo runInfo;
        std::string errorMsg = m_drp->beginrun(phase1Info, runInfo);
        if (!errorMsg.empty()) {
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }

        m_drp->runInfoData(xtc, bufEnd, m_det->namesLookup(), runInfo);
    }
    else if (key == "endrun") {
        std::string errorMsg = m_drp->endrun(phase1Info);
        if (!errorMsg.empty()) {
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }
    }
    else if (key == "enable") {
        bool chunkRequest;
        ChunkInfo chunkInfo;
        std::string errorMsg = m_drp->enable(phase1Info, chunkRequest, chunkInfo);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        } else if (chunkRequest) {
            logging::debug("handlePhase1 enable found chunkRequest");
            m_drp->chunkInfoData(xtc, bufEnd, m_det->namesLookup(), chunkInfo);
        }
        unsigned error = m_det->enable(xtc, bufEnd, phase1Info);
        if (error) {
            std::string errorMsg = "Phase 1 error in Detector::enable()";
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
        logging::debug("handlePhase1 enable complete");
    }

    json answer = createMsg(key, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void EpicsArchApp::handleReset(const json& msg)
{
    unsubscribePartition();             // ZMQ_UNSUBSCRIBE
    _unconfigure();
    _disconnect();
    connectionShutdown();
}

} // namespace Drp


static void usage(const char* name)
{
    printf( "Usage:  %s  [OPTIONS] <configuration filename>\n"
      "  Options:\n"
      "    -C      [*required*] Collection server\n"
      "    -d      [*required*] PGP device name\n"
      "    -u      [*required*] Process alias\n"
      "    -P      [*required*] Instrument name\n"
      "    -p      [*required*] Set partition id\n"
      "    -o      Set output file directory\n"
      "    -l      Set the PGP lane mask\n"
      "    -k      Option for supplying kwargs\n"
      "    -M      Prometheus config file directory\n"
      "    -T      Transition buffer size\n"
      "    -v      Verbosity level (repeat for increased detail)\n"
      "    -h      Show usage\n"
      "================================================================================\n"
      "  Config File Format:\n"
      "    - Each line of the file can contain one PV name\n"
      "    - Optionally follow the PV name with \'ca\' or \'pva\' to specify\n"
      "      the provider type (defaults to \'ca\')\n"
      "    - Use \'#\' at the beginning of the line to comment out whole line\n"
      "    - Use \'#\' in the middle of the line to comment out the remaining characters\n"
      "    - Use '*' at the beginning of the line to define an alias for the\n"
      "      immediately following PV(s).  This must be a valid Python name\n"
      "    - Use \'<\' to include file(s)\n"
      "  \n"
      "  Example:\n"
      "    %%cat epicsArch.txt\n"
      "    < PvList0.txt, PvList1.txt # Include Two Files\n"
      "    iocTest:aiExample          # PV Name, CA provider\n"
      "    # This is a comment line\n"
      "    iocTest:calcExample1 pva   # PV name, PVA provider\n"
      "    * electron_beam_energy     # Alias for BEND:DMP1:400:BDES\n"
      "    BEND:DMP1:400:BDES   ca    # PV name, CA provider\n",
      name
    );
}

int main(int argc, char* argv[])
{
    Drp::Parameters para;
    para.maxTrSize = 1024 * 1024;

    std::string kwargs_str;
    int c;
    while((c = getopt(argc, argv, "p:o:l:C:d:u:k:P:M:T:vh")) != EOF) {
        switch(c) {
            case 'p':
                para.partition = std::stoi(optarg);
                break;
            case 'o':
                para.outputDir = optarg;
                break;
            case 'l':
                para.laneMask = std::stoul(optarg, nullptr, 16);
                break;
            case 'C':
                para.collectionHost = optarg;
                break;
            case 'd':
                para.device = optarg;
                break;
            case 'u':
                para.alias = optarg;
                break;
            case 'k':
                kwargs_str = kwargs_str.empty()
                           ? optarg
                           : kwargs_str + "," + optarg;
                break;
            case 'P':
                para.instrument = optarg;
                break;
            case 'M':
                para.prometheusDir = optarg;
                break;
            case 'T':
                para.maxTrSize  = std::stoul(optarg, nullptr, 0);
                break;
            case 'v':
                ++para.verbose;
                break;
            case 'h':
            default:
                usage(argv[0]);
                return 1;
        }
    }

    switch (para.verbose) {
        case 0:  logging::init(para.instrument.c_str(), LOG_INFO);     break;
        default: logging::init(para.instrument.c_str(), LOG_DEBUG);    break;
    }
    logging::info("logging configured");
    if (para.instrument.empty()) {
        logging::warning("-P: instrument name is missing");
    }
    // Check required parameters
    if (para.partition == unsigned(-1)) {
        logging::critical("-p: partition is mandatory");
        return 1;
    }
    if (para.device.empty()) {
        logging::critical("-d: device is mandatory");
        return 1;
    }
    if (para.alias.empty()) {
        logging::critical("-u: alias is mandatory");
        return 1;
    }

    // Only one lane is supported by this DRP
    if (std::bitset<8>(para.laneMask).count() != 1) {
        logging::critical("-l: lane mask must have only 1 bit set");
        return 1;
    }

    // Alias must be of form <detName>_<detSegment>
    size_t found = para.alias.rfind('_');
    if ((found == std::string::npos) || !isdigit(para.alias.back())) {
        logging::critical("-u: alias must have _N suffix");
        return 1;
    }
    para.detType = "epics";
    para.detName = "epics";  //para.alias.substr(0, found);
    para.detSegment = std::stoi(para.alias.substr(found+1, para.alias.size()));
    para.serNo = "detnum1234";

    std::string pvCfgFile;
    if (optind < argc)
    {
        pvCfgFile = argv[optind++];

        if (optind < argc)
        {
            logging::error("Unrecognized argument:");
            while (optind < argc)
                logging::error("  %s ", argv[optind++]);
            return 1;
        }
    }
    else {
        logging::critical("A PV config filename is mandatory");
        return 1;
    }

    try {
        get_kwargs(kwargs_str, para.kwargs);
        for (const auto& kwargs : para.kwargs)
        {
            if (kwargs.first == "forceEnet")         continue;
            if (kwargs.first == "ep_fabric")         continue;
            if (kwargs.first == "ep_domain")         continue;
            if (kwargs.first == "ep_provider")       continue;
            if (kwargs.first == "sim_length")        continue;  // XpmDetector
            if (kwargs.first == "timebase")          continue;  // XpmDetector
            if (kwargs.first == "pebbleBufSize")     continue;  // DrpBase
            if (kwargs.first == "pebbleBufCount")    continue;  // DrpBase
            if (kwargs.first == "batching")          continue;  // DrpBase
            if (kwargs.first == "directIO")          continue;  // DrpBase
            if (kwargs.first == "pva_addr")          continue;  // DrpBase
            logging::critical("Unrecognized kwarg '%s=%s'\n",
                              kwargs.first.c_str(), kwargs.second.c_str());
            return 1;
        }

        Drp::EpicsArchApp app(para, pvCfgFile);
        app.run();
        return 0;
    }
    catch (std::exception& e)  { logging::critical("%s", e.what()); }
    catch (std::string& e)     { logging::critical("%s", e.c_str()); }
    catch (char const* e)      { logging::critical("%s", e); }
    catch (...)                { logging::critical("Default exception"); }
    return EXIT_FAILURE;
}
