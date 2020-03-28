#include "EpicsArch.hh"

#include <cassert>
#include <chrono>
#include <unistd.h>
#include <iostream>
#include <map>
#include <getopt.h>
#include <Python.h>
#include "DataDriver.h"
#include "drp/RunInfoDef.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psalg/utils/SysLog.hh"


using json = nlohmann::json;
using logging = psalg::SysLog;
using namespace Pds;

namespace Drp {

class Pgp
{
public:
    Pgp(const Parameters& para, DrpBase& drp, const bool& running) :
        m_para(para), m_pool(drp.pool), m_tebContributor(drp.tebContributor()), m_running(running),
        m_available(0), m_current(0), m_lastComplete(0)
    {
        m_nodeId = drp.nodeId();
        uint8_t mask[DMA_MASK_SIZE];
        dmaInitMaskBytes(mask);
        for (unsigned i=0; i<4; i++) {
            if (para.laneMask & (1 << i)) {
                logging::info("setting lane  %d", i);
                dmaAddMaskBytes((uint8_t*)mask, dmaDest(i, 0));
            }
        }
        dmaSetMaskBytes(drp.pool.fd(), mask);
    }

    Pds::EbDgram* next(uint32_t& evtIndex, uint64_t& bytes);
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
};

Pds::EbDgram* Pgp::_handle(uint32_t& current, uint64_t& bytes)
{
    int32_t size = dmaRet[m_current];
    uint32_t index = dmaIndex[m_current];
    uint32_t lane = (dest[m_current] >> 8) & 7;
    bytes += size;
    if (unsigned(size) > m_pool.dmaSize()) {
        logging::critical("DMA overflowed buffer: %d vs %d\n", size, m_pool.dmaSize());
        exit(-1);
    }

    const uint32_t* data = (uint32_t*)m_pool.dmaBuffers[index];
    uint32_t evtCounter = data[5] & 0xffffff;
    const unsigned bufferMask = m_pool.nbuffers() - 1;
    current = evtCounter & bufferMask;
    PGPEvent* event = &m_pool.pgpEvents[current];

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
        logging::debug("PGPReader  saw %s transition @ %u.%09u (%014lx)",
                       XtcData::TransitionId::name(transitionId),
                       timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                       timingHeader->pulseId());
    }
    if (evtCounter != ((m_lastComplete + 1) & 0xffffff)) {
        logging::critical("%PGPReader: Jump in complete l1Count %u -> %u | difference %d, tid %s%s",
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

    event->l3InpBuf = m_tebContributor.allocate(*timingHeader, (void*)((uintptr_t)current));

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
            if (m_available > 0) {
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

unsigned EaDetector::configure(const std::string& config_alias, XtcData::Xtc& xtc)
{
    logging::info("EpicsArch configure");

    m_exporter = std::make_shared<MetricExporter>();
    if (m_drp.exposer()) {
        m_drp.exposer()->RegisterCollectable(m_exporter);
    }

    try
    {
        std::string configFileWarning;
        m_monitor = std::make_unique<EpicsArchMonitor>(m_pvCfgFile.c_str(), m_para->verbose, configFileWarning);
        if (!configFileWarning.empty()) {
            logging::warning("%s", configFileWarning.c_str());
        }
    }
    catch(std::string& error)
    {
        logging::error("%s: new EpicsArchMonitor( %s ) failed: %s\n",
                       __PRETTY_FUNCTION__, m_pvCfgFile.c_str(), error.c_str());
        m_monitor.reset();
        return 1;
    }

    // Wait for PVs to connect or be timed out
    unsigned pvCount = 0;
    unsigned nNotConnected = m_monitor->validate(pvCount);
    if (nNotConnected) {
        logging::error("Number of PVs that didn't connect: %d (of %d)", nNotConnected, pvCount);
        return 1;
    }

    size_t payloadSize;
    m_monitor->initDef(payloadSize);
    logging::debug("payloadSize %zd", payloadSize);
    if (payloadSize > m_pool->bufferSize()) {
        logging::error("Event buffer size (%zd) is too small for payload (%zd)",
                       m_pool->bufferSize(), payloadSize);
        return 1;
    }

    m_monitor->addNames(m_para->detName, m_para->detType, m_para->serNo,
                        xtc, m_namesLookup, nodeId);

    m_workerThread = std::thread{&EaDetector::_worker, this};

    return 0;
}

void EaDetector::event(XtcData::Dgram& dgram, PGPEvent* event)
{
    m_monitor->getData(dgram.xtc, m_namesLookup, nodeId);
}

void EaDetector::shutdown()
{
    m_exporter.reset();

    m_terminate.store(true, std::memory_order_release);
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }
    m_monitor.reset();
    m_namesLookup.clear();   // erase all elements
}

void EaDetector::_worker()
{
    // setup monitoring
    std::map<std::string, std::string> labels{{"partition", std::to_string(m_para->partition)}};
    m_nEvents = 0;
    m_exporter->add("drp_event_rate", labels, MetricType::Rate,
                    [&](){return m_nEvents;});
    uint64_t bytes = 0L;
    m_exporter->add("drp_pgp_byte_rate", labels, MetricType::Rate,
                    [&](){return bytes;});
    m_nUpdates = 0;
    m_exporter->add("ea_update_count", labels, MetricType::Counter,
                    [&](){return m_nUpdates;});

    Pgp pgp(*m_para, m_drp, m_running);

    m_terminate.store(false, std::memory_order_release);

    while (true) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }

        uint32_t index;
        Pds::EbDgram* dgram = pgp.next(index, bytes);
        if (dgram) {
            XtcData::TransitionId::Value service = dgram->service();
            logging::debug("EAWorker saw %s transition @ %d.%09d (%014lx)\n",
                           XtcData::TransitionId::name(service),
                           dgram->time.seconds(), dgram->time.nanoseconds(), dgram->pulseId());
            if (service != XtcData::TransitionId::L1Accept) {
                // Allocate a transition dgram from the pool and initialize its header
                Pds::EbDgram* trDgram = m_pool->allocateTr();
                *trDgram = *dgram;
                PGPEvent* pgpEvent = &m_pool->pgpEvents[index];
                pgpEvent->transitionDgram = trDgram;

                if (service == XtcData::TransitionId::SlowUpdate) {
                    m_nUpdates++;

                    event(*trDgram, pgpEvent);
                }
                else {
                    // copy the temporary xtc created on phase 1 of the transition
                    // into the real location
                    XtcData::Xtc& trXtc = transitionXtc();
                    memcpy(&trDgram->xtc, &trXtc, trXtc.extent);

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
    m_nEvents++;

    // Make sure the datagram didn't get too big
    const size_t size = sizeof(dgram) + dgram.xtc.sizeofPayload();
    const size_t maxSize = ((dgram.service() == XtcData::TransitionId::L1Accept) ||
                            (dgram.service() == XtcData::TransitionId::SlowUpdate))
                         ? m_pool->bufferSize()
                         : m_para->maxTrSize;
    if (size > maxSize) {
        logging::critical("%s Dgram of size %zd overflowed buffer of size %zd", XtcData::TransitionId::name(dgram.service()), size, maxSize);
        exit(-1);
    }

    PGPEvent* event = &m_drp.pool.pgpEvents[index];
    if (event->l3InpBuf) { // else timed out
        Pds::EbDgram* l3InpDg = new(event->l3InpBuf) Pds::EbDgram(dgram);
        if (dgram.isEvent()) {
            if (m_drp.triggerPrimitive()) { // else this DRP doesn't provide input
                m_drp.triggerPrimitive()->event(m_drp.pool, index, dgram.xtc, l3InpDg->xtc); // Produce
            }
        }
        m_drp.tebContributor().process(l3InpDg);
    }
}


EpicsArchApp::EpicsArchApp(Drp::Parameters& para, const std::string& pvCfgFile) :
    CollectionApp(para.collectionHost, para.partition, "drp", para.alias),
    m_drp(para, context()),
    m_para(para),
    m_det(std::make_unique<EaDetector>(m_para, pvCfgFile, m_drp))
{
    if (m_det == nullptr) {
        logging::critical("Error !! Could not create Detector object");
        throw "Could not create Detector object";
    }
    if (m_para.outputDir.empty()) {
        logging::info("output dir: n/a");
    } else {
        logging::info("output dir: %s", m_para.outputDir.c_str());
    }
    logging::info("Ready for transitions");
}

void EpicsArchApp::_shutdown()
{
    m_drp.shutdown();        // TebContributor must be shut down befoe the worker
    m_det->shutdown();
}

json EpicsArchApp::connectionInfo()
{
    std::string ip = getNicIp();
    logging::debug("nic ip  %s", ip.c_str());
    json body = {{"connect_info", {{"nic_ip", ip}}}};
    json info = m_det->connectionInfo();
    body["connect_info"].update(info);
    json bufInfo = m_drp.connectionInfo();
    body["connect_info"].update(bufInfo);
    return body;
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

    m_unconfigure = false;

    json body = json({});
    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void EpicsArchApp::handleDisconnect(const json& msg)
{
    _shutdown();

    json body = json({});
    reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void EpicsArchApp::handlePhase1(const json& msg)
{
    std::string key = msg["header"]["key"];
    logging::debug("handlePhase1 for %s in EpicsArchApp", key.c_str());

    XtcData::Xtc& xtc = m_det->transitionXtc();
    XtcData::TypeId tid(XtcData::TypeId::Parent, 0);
    xtc.src = XtcData::Src(m_det->nodeId); // set the src field for the event builders
    xtc.damage = 0;
    xtc.contains = tid;
    xtc.extent = sizeof(XtcData::Xtc);

    json phase1Info{ "" };
    if (msg.find("body") != msg.end()) {
        if (msg["body"].find("phase1Info") != msg["body"].end()) {
            phase1Info = msg["body"]["phase1Info"];
        }
    }

    json body = json({});

    if (key == "configure") {
        if (m_unconfigure) {
            _shutdown();
            m_unconfigure = false;
        }

        std::string errorMsg = m_drp.configure(msg);
        if (!errorMsg.empty()) {
            errorMsg = "Phase 1 error: " + errorMsg;
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }
        else {
            std::string config_alias = msg["body"]["config_alias"];
            unsigned error = m_det->configure(config_alias, xtc);
            if (error) {
                std::string errorMsg = "Phase 1 error in Detector::configure";
                logging::error("%s", errorMsg.c_str());
                _error(key, msg, errorMsg);
                return;
            }
            else {
                m_drp.runInfoSupport(xtc, m_det->namesLookup());
            }
        }
    }
    else if (key == "unconfigure") {
        // Delay unconfiguration until after phase 2 of unconfigure has completed
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
            m_drp.runInfoData(xtc, m_det->namesLookup(), runInfo);
        }
    }
    else if (key == "endrun") {
        std::string errorMsg = m_drp.endrun(phase1Info);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
    }

    json answer = createMsg(key, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void EpicsArchApp::handleReset(const nlohmann::json& msg)
{
    _shutdown();
}

} // namespace Drp


void get_kwargs(Drp::Parameters& para, const std::string& kwargs_str) {
    std::istringstream ss(kwargs_str);
    std::string kwarg;
    std::string::size_type pos = 0;
    while (getline(ss, kwarg, ',')) {
        pos = kwarg.find("=", pos);
        if (!pos) {
          logging::critical("Keyword argument with no equal sign");
            throw "error: keyword argument with no equal sign: "+kwargs_str;
        }
        std::string key = kwarg.substr(0,pos);
        std::string value = kwarg.substr(pos+1,kwarg.length());
        //cout << kwarg << " " << key << " " << value << endl;
        para.kwargs[key] = value;
    }
}

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
      "    -T      ConfigDb detName for trigger\n"
      "            (-T without arg gives system default;\n"
      "             n.b. no space between -T and arg)\n"
      "    -M      Prometheus config file directory\n"
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
      "      immediately following PV(s)\n"
      "    - Use \'<\' to include file(s)\n"
      "  \n"
      "  Example:\n"
      "    %%cat epicsArch.txt\n"
      "    < PvList0.txt, PvList1.txt # Include Two Files\n"
      "    iocTest:aiExample          # PV Name, CA provider\n"
      "    # This is a comment line\n"
      "    iocTest:calcExample1 pva   # PV name, PVA provider\n"
      "    * electron beam energy     # Alias for BEND:DMP1:400:BDES\n"
      "    BEND:DMP1:400:BDES   ca    # PV name, CA provider\n",
      name
    );
}

int main(int argc, char* argv[])
{
    Drp::Parameters para;
    std::string kwargs_str;
    int c;
    while((c = getopt(argc, argv, "p:o:C:d:u:k:P:T::M:vh")) != EOF) {
        switch(c) {
            case 'p':
                para.partition = std::stoi(optarg);
                break;
            case 'o':
                para.outputDir = optarg;
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
                kwargs_str = std::string(optarg);
                break;
            case 'P':
                para.instrument = optarg;
                break;
            case 'T':
                para.trgDetName = optarg ? optarg : "trigger";
                break;
            case 'M':
                para.prometheusDir = optarg;
                break;
            case 'v':
                ++para.verbose;
                break;
            case 'h':
            default:
                usage(argv[0]);
                exit(1);
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
        exit(1);
    }
    if (para.device.empty()) {
        logging::critical("-d: device is mandatory");
        exit(1);
    }
    if (para.alias.empty()) {
        logging::critical("-u: alias is mandatory");
        exit(1);
    }

    // Alias must be of form <detName>_<detSegment>
    size_t found = para.alias.rfind('_');
    if ((found == std::string::npos) || !isdigit(para.alias.back())) {
        logging::critical("-u: alias must have _N suffix");
        exit(1);
    }
    para.detName = para.alias.substr(0, found);
    para.detSegment = std::stoi(para.alias.substr(found+1, para.alias.size()));

    get_kwargs(para, kwargs_str);

    std::string pvCfgFile;
    if (optind < argc)
        pvCfgFile = argv[optind];
    else {
        logging::critical("A PV config filename is mandatory");
        exit(1);
    }

    para.laneMask = 0x1;
    para.detName = "epics";
    para.detType = "epics";
    para.maxTrSize = 256 * 1024;
    para.nTrBuffers = 8; // Power of 2 greater than the maximum number of
                         // transitions in the system at any given time, e.g.,
                         // MAX_LATENCY * (SlowUpdate rate), in same units

    Py_Initialize(); // for use by configuration
    Drp::EpicsArchApp app(para, pvCfgFile);
    app.run();
    app.handleReset(json({}));
    Py_Finalize(); // for use by configuration
}
