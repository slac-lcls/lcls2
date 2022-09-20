#include "EpicsArch.hh"

#include <cassert>
#include <bitset>
#include <chrono>
#include <unistd.h>
#include <iostream>
#include <map>
#include <algorithm>
#include <getopt.h>
#include <Python.h>
#include "DataDriver.h"
#include "drp/RunInfoDef.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psalg/utils/SysLog.hh"

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

using json = nlohmann::json;
using logging = psalg::SysLog;
using ms_t = std::chrono::milliseconds;

namespace Drp {

class Pgp
{
public:
    Pgp(const Parameters& para, DrpBase& drp, const bool& running) :
        m_para(para), m_pool(drp.pool), m_tebContributor(drp.tebContributor()), m_running(running),
        m_available(0), m_current(0), m_lastComplete(0), m_latency(0), m_nDmaRet(0)
    {
        m_nodeId = drp.nodeId();
        if (drp.pool.setMaskBytes(para.laneMask, 0)) {
            logging::error("Failed to allocate lane/vc");
        }
    }

    Pds::EbDgram* next(uint32_t& evtIndex, uint64_t& bytes);
    const int64_t latency() { return m_latency; }
    const uint64_t nDmaRet() { return m_nDmaRet; }
private:
    Pds::EbDgram* _handle(uint32_t& evtIndex, uint64_t& bytes);
    const Parameters& m_para;
    MemPool& m_pool;
    Pds::Eb::TebContributor& m_tebContributor;
    static const int MAX_RET_CNT_C = 100;
    int32_t dmaRet[MAX_RET_CNT_C];
    uint32_t dmaIndex[MAX_RET_CNT_C];
    uint32_t dest[MAX_RET_CNT_C];
    const bool& m_running;
    int32_t m_available;
    int32_t m_current;
    uint32_t m_lastComplete;
    XtcData::TransitionId::Value m_lastTid;
    uint32_t m_lastData[6];
    unsigned m_nodeId;
    int64_t m_latency;
    uint64_t m_nDmaRet;
};

Pds::EbDgram* Pgp::_handle(uint32_t& current, uint64_t& bytes)
{
    uint32_t size = dmaRet[m_current];
    uint32_t index = dmaIndex[m_current];
    uint32_t lane = (dest[m_current] >> 8) & 7;
    bytes += size;
    if (size > m_pool.dmaSize()) {
        logging::critical("DMA overflowed buffer: %d vs %d", size, m_pool.dmaSize());
        throw "DMA overflowed buffer";
    }

    const uint32_t* data = (uint32_t*)m_pool.dmaBuffers[index];
    uint32_t evtCounter = data[5] & 0xffffff;
    const unsigned bufferMask = m_pool.nbuffers() - 1;
    current = evtCounter & bufferMask;
    PGPEvent* event = &m_pool.pgpEvents[current];
    // Revisit: Doesn't always work?  assert(event->mask == 0);

    DmaBuffer* buffer = &event->buffers[lane];
    buffer->size = size;
    buffer->index = index;
    event->mask |= (1 << lane);

    logging::debug("PGPReader  lane %d  size %d  hdr %016lx.%016lx.%08x",
                   lane, size,
                   reinterpret_cast<const uint64_t*>(data)[0],
                   reinterpret_cast<const uint64_t*>(data)[1],
                   reinterpret_cast<const uint32_t*>(data)[4]);

    const Pds::TimingHeader* timingHeader = reinterpret_cast<const Pds::TimingHeader*>(data);
    if (timingHeader->error()) {
        logging::error("Timing header error bit is set");
    }
    XtcData::TransitionId::Value transitionId = timingHeader->service();
    if (transitionId != XtcData::TransitionId::L1Accept) {
        if (transitionId != XtcData::TransitionId::SlowUpdate) {
            logging::info("PGPReader  saw %s @ %u.%09u (%014lx)",
                          XtcData::TransitionId::name(transitionId),
                          timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                          timingHeader->pulseId());
        }
        else {
            logging::debug("PGPReader  saw %s @ %u.%09u (%014lx)",
                           XtcData::TransitionId::name(transitionId),
                           timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                           timingHeader->pulseId());
        }
        if (transitionId == XtcData::TransitionId::BeginRun) {
            m_lastComplete = 0;  // EvtCounter reset
        }
    }
    if (evtCounter != ((m_lastComplete + 1) & 0xffffff)) {
        logging::critical("%sPGPReader: Jump in complete l1Count %u -> %u | difference %d, tid %s%s",
                          RED_ON, m_lastComplete, evtCounter, evtCounter - m_lastComplete, XtcData::TransitionId::name(transitionId), RED_OFF);
        logging::critical("data: %08x %08x %08x %08x %08x %08x",
                          data[0], data[1], data[2], data[3], data[4], data[5]);

        logging::critical("lastTid %s", XtcData::TransitionId::name(m_lastTid));
        logging::critical("lastData: %08x %08x %08x %08x %08x %08x",
                          m_lastData[0], m_lastData[1], m_lastData[2], m_lastData[3], m_lastData[4], m_lastData[5]);

        throw "Jump in event counter";

        for (unsigned e=m_lastComplete+1; e<evtCounter; e++) {
            PGPEvent* brokenEvent = &m_pool.pgpEvents[e & bufferMask];
            logging::error("broken event:  %08x", brokenEvent->mask);
            brokenEvent->mask = 0;

        }
    }
    m_lastComplete = evtCounter;
    m_lastTid = transitionId;
    memcpy(m_lastData, data, 24);

    auto now = std::chrono::system_clock::now();
    auto dgt = std::chrono::seconds{timingHeader->time.seconds() + POSIX_TIME_AT_EPICS_EPOCH}
             + std::chrono::nanoseconds{timingHeader->time.nanoseconds()};
    std::chrono::system_clock::time_point tp{std::chrono::duration_cast<std::chrono::system_clock::duration>(dgt)};
    m_latency = std::chrono::duration_cast<ms_t>(now - tp).count();

    // make new dgram in the pebble
    // It must be an EbDgram in order to be able to send it to the MEB
    Pds::EbDgram* dgram = new(m_pool.pebble[current]) Pds::EbDgram(*timingHeader, XtcData::Src(m_nodeId), m_para.rogMask);

    return dgram;
}

Pds::EbDgram* Pgp::next(uint32_t& evtIndex, uint64_t& bytes)
{
    // get new buffers
    if (m_current == m_available) {
        m_current = 0;
        auto start = std::chrono::steady_clock::now();
        while (true) {
            m_available = dmaReadBulkIndex(m_pool.fd(), MAX_RET_CNT_C, dmaRet, dmaIndex, NULL, NULL, dest);
            m_nDmaRet = m_available;
            if (m_available > 0) {
                m_pool.allocate(m_available);
                break;
            }

            // wait for a total of 10 ms otherwise timeout
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (elapsed > 10) {
                //if (m_running)  logging::debug("pgp timeout");
                return nullptr;
            }
        }
    }

    Pds::EbDgram* dgram = _handle(evtIndex, bytes);
    m_current++;
    return dgram;
}


EaDetector::EaDetector(Parameters& para, const std::string& pvCfgFile, DrpBase& drp) :
    XpmDetector(&para, &drp.pool),
    m_pvCfgFile(pvCfgFile),
    m_drp(drp),
    m_terminate(false),
    m_running(false)
{
}

EaDetector::~EaDetector()
{
    EpicsArchMonitor::close();
}

unsigned EaDetector::connect(std::string& msg)
{
    try
    {
        std::string configFileWarning;
        m_monitor = std::make_unique<EpicsArchMonitor>(m_pvCfgFile.c_str(), m_para->verbose, configFileWarning);
        if (!configFileWarning.empty()) {
            msg = configFileWarning;
        }
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
    m_monitor.reset();
    return 0;
}

unsigned EaDetector::configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd)
{
    logging::info("EpicsArch configure");

    if (XpmDetector::configure(config_alias, xtc, bufEnd))
        return 1;

    if (m_exporter)  m_exporter.reset();
    m_exporter = std::make_shared<Pds::MetricExporter>();
    if (m_drp.exposer()) {
        m_drp.exposer()->RegisterCollectable(m_exporter);
    }

    size_t payloadSize;
    m_monitor->addNames(m_para->detName, m_para->detType, m_para->serNo,
                        m_para->detSegment,
                        xtc, bufEnd, m_namesLookup, nodeId, payloadSize);
    // make sure config transition will fit in the transition buffer (where SlowUpdate goes), however this check is too late: code can unfortunately segfault in the above line -cpo
    if (sizeof(XtcData::Dgram)+xtc.sizeofPayload() > m_para->maxTrSize) {
        logging::critical("Increase Parameter::maxTrSize (%zd) to avoid truncation of configure transition (%zd)",
                          m_para->maxTrSize, sizeof(XtcData::Dgram)+xtc.sizeofPayload());
        abort();
    }
    // make sure the data will fit in the transition buffer (where SlowUpdate goes)
    if (payloadSize > m_para->maxTrSize) {
        logging::critical("Increase Parameter::maxTrSize (%zd) to avoid truncation of SlowUpdate data (%zd)",
                         m_para->maxTrSize, payloadSize);
        abort();
    }

    m_workerThread = std::thread{&EaDetector::_worker, this};

    return 0;
}

unsigned EaDetector::unconfigure()
{
    m_terminate.store(true, std::memory_order_release);
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }
    m_exporter.reset();
    m_namesLookup.clear();   // erase all elements

    return 0;
}

void EaDetector::event(XtcData::Dgram&, const void* bufEnd, PGPEvent*)
{
    // Unused
}

void EaDetector::slowupdate(XtcData::Xtc& xtc, const void* bufEnd)
{
    m_monitor->getData(xtc, bufEnd, m_namesLookup, nodeId, m_nStales);
}

void EaDetector::_worker()
{
    // setup monitoring
    std::map<std::string, std::string> labels{{"instrument", m_para->instrument},
                                              {"partition", std::to_string(m_para->partition)},
                                              {"detname", m_para->detName},
                                              {"detseg", std::to_string(m_para->detSegment)},
                                              {"alias", m_para->alias}};
    m_nEvents = 0;
    m_exporter->add("drp_event_rate", labels, Pds::MetricType::Rate,
                    [&](){return m_nEvents;});
    uint64_t bytes = 0L;
    m_exporter->add("drp_pgp_byte_rate", labels, Pds::MetricType::Rate,
                    [&](){return bytes;});
    m_nUpdates = 0;
    m_exporter->add("ea_update_count", labels, Pds::MetricType::Counter,
                    [&](){return m_nUpdates;});
    m_nStales = 0;
    m_exporter->add("ea_stale_count", labels, Pds::MetricType::Counter,
                    [&](){return m_nStales;});

    Pgp pgp(*m_para, m_drp, m_running);

    m_exporter->add("drp_th_latency", labels, Pds::MetricType::Gauge,
                    [&](){return pgp.latency();});
    m_exporter->add("drp_num_dma_ret", labels, Pds::MetricType::Gauge,
                    [&](){return pgp.nDmaRet();});

    m_terminate.store(false, std::memory_order_release);

    while (true) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }

        uint32_t index;
        Pds::EbDgram* dgram = pgp.next(index, bytes);
        if (dgram) {
            m_nEvents++;

            XtcData::TransitionId::Value service = dgram->service();
            logging::debug("EAWorker saw %s @ %d.%09d (%014lx)",
                           XtcData::TransitionId::name(service),
                           dgram->time.seconds(), dgram->time.nanoseconds(), dgram->pulseId());
            if (service != XtcData::TransitionId::L1Accept) {
                // Allocate a transition dgram from the pool and initialize its header
                Pds::EbDgram* trDgram = m_pool->allocateTr();
                const void*   bufEnd  = (char*)trDgram + m_para->maxTrSize;
                *trDgram = *dgram;
                PGPEvent* pgpEvent = &m_pool->pgpEvents[index];
                pgpEvent->transitionDgram = trDgram;

                if (service == XtcData::TransitionId::SlowUpdate) {
                    m_nUpdates++;

                    slowupdate(trDgram->xtc, bufEnd);
                }
                else {
                    // copy the temporary xtc created on phase 1 of the transition
                    // into the real location
                    XtcData::Xtc& trXtc = transitionXtc();
                    trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
                    auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
                    memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());

                    if (service == XtcData::TransitionId::Enable) {
                        m_running = true;
                    }
                    else if (service == XtcData::TransitionId::Disable) {
                        m_running = false;
                    }
                }
            }

            _sendToTeb(*dgram, index);
        }
    }
    logging::info("Worker thread finished");
}

void EaDetector::_sendToTeb(Pds::EbDgram& dgram, uint32_t index)
{
    // Make sure the datagram didn't get too big
    const size_t size = sizeof(dgram) + dgram.xtc.sizeofPayload();
    const size_t maxSize = (dgram.service() == XtcData::TransitionId::L1Accept)
                         ? m_pool->pebble.bufferSize()
                         : m_para->maxTrSize;
    if (size > maxSize) {
        logging::critical("%s Dgram of size %zd overflowed buffer of size %zd", XtcData::TransitionId::name(dgram.service()), size, maxSize);
        throw "Dgram overflowed buffer";
    }

    auto l3InpBuf = m_drp.tebContributor().fetch(index);
    Pds::EbDgram* l3InpDg = new(l3InpBuf) Pds::EbDgram(dgram);
    if (l3InpDg->isEvent()) {
        auto triggerPrimitive = m_drp.triggerPrimitive();
        if (triggerPrimitive) { // else this DRP doesn't provide input
            const void* bufEnd = (char*)l3InpDg + sizeof(*l3InpDg) + triggerPrimitive->size();
            triggerPrimitive->event(m_drp.pool, index, dgram.xtc, l3InpDg->xtc, bufEnd); // Produce
        }
    }
    m_drp.tebContributor().process(l3InpDg);
}


EpicsArchApp::EpicsArchApp(Drp::Parameters& para, const std::string& pvCfgFile) :
    CollectionApp(para.collectionHost, para.partition, "drp", para.alias),
    m_drp        (para, context()),
    m_para       (para),
    m_eaDetector (std::make_unique<EaDetector>(m_para, pvCfgFile, m_drp)),
    m_det        (m_eaDetector.get()),
    m_unconfigure(false)
{
    Py_Initialize();                    // for use by configuration

    if (m_det == nullptr) {
        logging::critical("Error !! Could not create Detector object for %s", m_para.detType.c_str());
        throw "Fatal: Could not create Detector object";
    }
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
    m_drp.disconnect();
    m_det->shutdown();
    m_eaDetector->disconnect();
}

void EpicsArchApp::_unconfigure()
{
    m_drp.pool.shutdown();  // Release Tr buffer pool
    m_drp.unconfigure();    // TebContributor must be shut down before the worker
    m_eaDetector->unconfigure();
    m_unconfigure = false;
}

json EpicsArchApp::connectionInfo()
{
    std::string ip = m_para.kwargs.find("ep_domain") != m_para.kwargs.end()
                   ? getNicIp(m_para.kwargs["ep_domain"])
                   : getNicIp(m_para.kwargs["forceEnet"] == "yes");
    logging::debug("nic ip  %s", ip.c_str());
    json body = {{"connect_info", {{"nic_ip", ip}}}};
    json info = m_det->connectionInfo();
    body["connect_info"].update(info);
    json bufInfo = m_drp.connectionInfo(ip);
    body["connect_info"].update(bufInfo);
    return body;
}

void EpicsArchApp::connectionShutdown()
{
    m_drp.shutdown();
}

void EpicsArchApp::_error(const std::string& which, const nlohmann::json& msg, const std::string& errorMsg)
{
    json body = json({});
    body["err_info"] = errorMsg;
    json answer = createMsg(which, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void EpicsArchApp::handleConnect(const nlohmann::json& msg)
{
    std::string errorMsg = m_drp.connect(msg, getId());
    if (!errorMsg.empty()) {
        logging::error("Error in DrpBase::connect");
        logging::error("%s", errorMsg.c_str());
        _error("connect", msg, errorMsg);
        return;
    }

    m_det->nodeId = m_drp.nodeId();
    m_det->connect(msg, std::to_string(getId()));

    unsigned rc = m_eaDetector->connect(errorMsg);
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

    XtcData::Xtc& xtc = m_det->transitionXtc();
    xtc = {{XtcData::TypeId::Parent, 0}, {m_det->nodeId}};
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

        std::string errorMsg = m_drp.configure(msg);
        if (!errorMsg.empty()) {
            errorMsg = "Phase 1 error: " + errorMsg;
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }

        std::string config_alias = msg["body"]["config_alias"];
        unsigned error = m_det->configure(config_alias, xtc, bufEnd);
        if (error) {
            std::string errorMsg = "Failed transition phase 1";
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }

        m_drp.runInfoSupport(xtc, bufEnd, m_det->namesLookup());
        m_drp.chunkInfoSupport(xtc, bufEnd, m_det->namesLookup());
    }
    else if (key == "unconfigure") {
        // "Queue" unconfiguration until after phase 2 has completed
        m_unconfigure = true;
    }
    else if (key == "beginrun") {
        RunInfo runInfo;
        std::string errorMsg = m_drp.beginrun(phase1Info, runInfo);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
        else if (runInfo.runNumber > 0) {
            m_drp.runInfoData(xtc, bufEnd, m_det->namesLookup(), runInfo);
        }
    }
    else if (key == "endrun") {
        std::string errorMsg = m_drp.endrun(phase1Info);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
    }
    else if (key == "enable") {
        bool chunkRequest;
        ChunkInfo chunkInfo;
        std::string errorMsg = m_drp.enable(phase1Info, chunkRequest, chunkInfo);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        } else if (chunkRequest) {
            logging::debug("handlePhase1 enable found chunkRequest");
            m_drp.chunkInfoData(xtc, bufEnd, m_det->namesLookup(), chunkInfo);
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

void EpicsArchApp::handleReset(const nlohmann::json& msg)
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
                           : kwargs_str + ", " + optarg;
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
            if (kwargs.first == "batching")          continue;  // DrpBase
            if (kwargs.first == "directIO")          continue;  // DrpBase
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
