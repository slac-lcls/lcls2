#include "EpicsArch.hh"

#include <cassert>
#include <chrono>
#include <unistd.h>
#include <iostream>
#include "DataDriver.h"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "drp/RunInfoDef.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psalg/utils/SysLog.hh"
#include <getopt.h>
#include <Python.h>


using json = nlohmann::json;
using logging = psalg::SysLog;
using namespace Pds;

namespace Drp {

class Pgp
{
public:
    Pgp(MemPool& pool, Pds::Eb::TebContributor& tebContributor, unsigned nodeId, uint32_t envMask) :
        m_pool(pool), m_tebContributor(tebContributor), m_nodeId(nodeId), m_envMask(envMask), m_available(0), m_current(0)
    {
        uint8_t mask[DMA_MASK_SIZE];
        dmaInitMaskBytes(mask);
        for (unsigned i=0; i<4; i++) {
            dmaAddMaskBytes((uint8_t*)mask, dmaDest(i, 0));
        }
        dmaSetMaskBytes(pool.fd(), mask);
    }

    Pds::EbDgram* next(uint32_t& evtIndex);
private:
    Pds::EbDgram* _handle(const Pds::TimingHeader& timingHeader, uint32_t& evtIndex);
    MemPool& m_pool;
    Pds::Eb::TebContributor& m_tebContributor;
    unsigned m_nodeId;
    uint32_t m_envMask;
    int32_t m_available;
    int32_t m_current;
    static const int MAX_RET_CNT_C = 100;
    int32_t dmaRet[MAX_RET_CNT_C];
    uint32_t dmaIndex[MAX_RET_CNT_C];
    uint32_t dest[MAX_RET_CNT_C];
};

Pds::EbDgram* Pgp::_handle(const Pds::TimingHeader& timingHeader, uint32_t& evtIndex)
{
    int32_t size = dmaRet[m_current];
    uint32_t index = dmaIndex[m_current];
    uint32_t lane = (dest[m_current] >> 8) & 7;
    if (unsigned(size) > m_pool.dmaSize()) {
        logging::critical("DMA overflowed buffer: %d vs %d\n", size, m_pool.dmaSize());
        exit(-1);
    }

    const uint32_t* data = (uint32_t*)m_pool.dmaBuffers[index];
    uint32_t evtCounter = data[5] & 0xffffff;
    evtIndex = evtCounter & (m_pool.nbuffers() - 1);
    PGPEvent* event = &m_pool.pgpEvents[evtIndex];

    DmaBuffer* buffer = &event->buffers[lane];
    buffer->size = size;
    buffer->index = index;
    event->mask |= (1 << lane);

    event->l3InpBuf = m_tebContributor.allocate(timingHeader, (void*)((uintptr_t)evtIndex));

    // make new dgram in the pebble
    // It must be an EbDgram in order to be able to send it to the MEB
    Pds::EbDgram* dgram = new(m_pool.pebble[evtIndex]) Pds::EbDgram(timingHeader, XtcData::Src(m_nodeId), m_envMask);

    return dgram;
}

Pds::EbDgram* Pgp::next(uint32_t& evtIndex)
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
                // printf("pgp timeout\n");
                return nullptr;
            }
        }
    }

    const Pds::TimingHeader* timingHeader = reinterpret_cast<Pds::TimingHeader*>(m_pool.dmaBuffers[dmaIndex[m_current]]);

    Pds::EbDgram* dgram = _handle(*timingHeader, evtIndex);
    m_current++;
    return dgram;
}

EpicsArchApp::EpicsArchApp(Drp::Parameters& para, const std::string& pvCfgFile) :
    CollectionApp(para.collectionHost, para.partition, "drp", para.alias),
    m_drp(para, context()),
    m_para(para),
    m_pvCfgFile(pvCfgFile),
    m_terminate(false)
{
    logging::info("Ready for transitions");
}

EpicsArchApp::~EpicsArchApp()
{
    EpicsArchMonitor::close();
}

void EpicsArchApp::_shutdown()
{
    m_exporter.reset();

    m_terminate.store(true, std::memory_order_release);
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }
    m_monitor.reset();
    m_drp.shutdown();
    m_namesLookup.clear();   // erase all elements
}

json EpicsArchApp::connectionInfo()
{
    std::string ip = getNicIp();
    logging::debug("nic ip  %s", ip.c_str());
    json body = {{"connect_info", {{"nic_ip", ip}}}};
    json bufInfo = m_drp.connectionInfo();
    body["connect_info"].update(bufInfo); // Revisit: Should be in det_info
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

    try
    {
        std::string configFileWarning;
        m_monitor = std::make_unique<EpicsArchMonitor>(m_pvCfgFile.c_str(), m_para.verbose, configFileWarning);
        if (!configFileWarning.empty()) {
            logging::warning("%s", configFileWarning.c_str());
        }
    }
    catch(std::string& error)
    {
        logging::error("%s: new EpicsArchMonitor( %s ) failed: %s\n",
                       __PRETTY_FUNCTION__, m_pvCfgFile.c_str(), error.c_str());
        _error("connect", msg, error);
        m_monitor = NULL;
        return;
    }

    _connectPgp(msg, std::to_string(getId()));

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
    json phase1Info{ "" };
    if (msg.find("body") != msg.end()) {
        if (msg["body"].find("phase1Info") != msg["body"].end()) {
            phase1Info = msg["body"]["phase1Info"];
        }
    }

    json body = json({});
    std::string key = msg["header"]["key"];
    logging::debug("handlePhase1 for %s in EpicsArchApp", key.c_str());

    if (key == "configure") {
        if (m_unconfigure) {
            _shutdown();
            m_unconfigure = false;
        }

        std::string errorMsg = m_drp.configure(msg);
        if (!errorMsg.empty()) {
            errorMsg = "Phase 1 error: " + errorMsg;
            _error(key, msg, errorMsg);
            logging::error("%s", errorMsg.c_str());
        }
        else {
            m_exporter = std::make_shared<MetricExporter>();
            if (m_drp.exposer()) {
                m_drp.exposer()->RegisterCollectable(m_exporter);
            }

            // Wait for PVs to connect or be timed out
            unsigned pvCount = 0;
            unsigned nNotConnected = m_monitor->validate(pvCount);
            if (nNotConnected) {
                errorMsg = "Unconnected PV(s)";
                body["err_info"] = errorMsg;
                logging::error("Number of PVs that didn't connect: %d (of %d)", nNotConnected, pvCount);
            }
            else {
                m_terminate.store(false, std::memory_order_release);

                m_workerThread = std::thread{&EpicsArchApp::_worker, this, m_exporter};
            }
        }
    }
    else if (key == "unconfigure") {
        // Delay unconfiguration until after phase 2 of unconfigure has completed
        m_unconfigure = true;
    }
    else if (key == "beginrun") {
        std::string errorMsg = m_drp.beginrun(phase1Info, m_runInfo);
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

void EpicsArchApp::_connectPgp(const json& json, const std::string& collectionId)
{
    // FIXME not sure what the size should be since for this DRP we expect no PGP payload
    int length = 0;
    int links = m_para.laneMask;

    int fd = open(m_para.device.c_str(), O_RDWR);
    if (fd < 0) {
        logging::error("Error opening %s", m_para.device.c_str());
    }

    int readoutGroup = json["body"]["drp"][collectionId]["det_info"]["readout"];
    uint32_t v = ((readoutGroup&0xf)<<0) |
                  ((length&0xffffff)<<4) |
                  (links<<28);
    dmaWriteRegister(fd, 0x00a00000, v);
    uint32_t w;
    dmaReadRegister(fd, 0x00a00000, &w);
    logging::info("Configured readout group [%u], length [%u], links [%x]: [%x](%x)",
           readoutGroup, length, links, v, w);
    for (unsigned i=0; i<4; i++) {
        if (links&(1<<i)) {
            // this is the threshold to assert deadtime (high water mark) for every link
            // 0x1f00 corresponds to 0x1f free buffers
            dmaWriteRegister(fd, 0x00800084+32*i, 0x1f00);
        }
    }
    close(fd);
}

void EpicsArchApp::_worker(std::shared_ptr<MetricExporter> exporter)
{
    m_monitor->initDef();

    Pgp pgp(m_drp.pool, m_drp.tebContributor(), m_drp.nodeId(), m_para.rogMask);

    std::map<std::string, std::string> labels{{"partition", std::to_string(m_para.partition)}};
    m_nEvents = 0;
    exporter->add("drp_event_rate", labels, MetricType::Rate,
                  [&](){return m_nEvents;});
    m_nUpdates = 0;
    exporter->add("ea_update_rate", labels, MetricType::Rate,
                  [&](){return m_nUpdates;});
    m_nConnected = 0;
    exporter->add("ea_connected_count", labels, MetricType::Counter,
                  [&](){return m_nConnected;});

    while (true) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }

        // Drain PGP to avoid inducing backpressure
        uint32_t index;
        Pds::EbDgram* dgram = pgp.next(index);
        if (dgram) {
            XtcData::TransitionId::Value service = dgram->service();
            logging::debug("EAWorker saw %s transition @ %d.%09d (%014lx)\n",
                           XtcData::TransitionId::name(service),
                           dgram->time.seconds(), dgram->time.nanoseconds(), dgram->pulseId());
            if (service != XtcData::TransitionId::L1Accept) {
                // Construct the transition in its own buffer from the PGP Dgram
                Pds::EbDgram* trDgram = m_drp.pool.transitionDgram();
                *trDgram = *dgram;

                switch (service) {
                    case XtcData::TransitionId::Configure: {
                        logging::info("EpicsArch configure");

                        m_monitor->addNames(trDgram->xtc, m_namesLookup, m_drp.nodeId());

                        m_drp.runInfoSupport(trDgram->xtc, m_namesLookup);
                        break;
                    }
                    case XtcData::TransitionId::BeginRun: {
                        if (m_runInfo.runNumber > 0) {
                            m_drp.runInfoData(trDgram->xtc, m_namesLookup, m_runInfo);
                        }
                        break;
                    }
                    case XtcData::TransitionId::SlowUpdate: {
                        // SlowUpdates are not synchronous like other tranisitions,
                        // so treat them like L1Accepts (i.e., use dgram vs trDgram)
                        m_monitor->getData(dgram->xtc, m_namesLookup, m_drp.nodeId());

                        // Make sure the XTC didn't get too big
                        size_t size = sizeof(*dgram) + dgram->xtc.sizeofPayload();
                        if (size > m_drp.pool.pebble.bufferSize()) {
                            logging::critical("SlowUpdate: buffer size (%zd) too small for Dgram (%zd)", m_drp.pool.pebble.bufferSize(), size);
                            exit(-1);
                        }
                        break;
                    }
                    default: {              // Handle other transitions
                        break;
                    }
                }

                // Make sure the transition didn't get too big
                size_t size = sizeof(*trDgram) + trDgram->xtc.sizeofPayload();
                if (size > m_para.maxTrSize) {
                    logging::critical("Transition: buffer size (%zd) too small for Dgram (%zd)", m_para.maxTrSize, size);
                    exit(-1);
                }
            }
            _sendToTeb(*dgram, index);
            m_nEvents++;
        }
    }
    logging::info("Worker thread finished");
}

void EpicsArchApp::_sendToTeb(Pds::EbDgram& dgram, uint32_t index)
{
    PGPEvent* event = &m_drp.pool.pgpEvents[index];
    void* buffer = event->l3InpBuf;
    if (buffer) { // else timed out
        Pds::EbDgram* l3InpDg = new(buffer) Pds::EbDgram(dgram);
        if (dgram.isEvent()) {
            if (m_drp.triggerPrimitive()) { // else this DRP doesn't provide input
                m_drp.triggerPrimitive()->event(m_drp.pool, index, dgram.xtc, l3InpDg->xtc); // Produce
                size_t size = sizeof(*l3InpDg) + l3InpDg->xtc.sizeofPayload();
                if (size > m_drp.tebPrms().maxInputSize) {
                    logging::critical("L3 Input Dgram of size %zd overflowed buffer of size %zd", size, m_drp.tebPrms().maxInputSize);
                    exit(-1);
                }
            }
        }
        m_drp.tebContributor().process(l3InpDg);
    }
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
    para.partition = -1;
    para.laneMask = 0x1;
    para.detName = "pva";               // Revisit: Should come from alias?
    para.detSegment = 0;
    para.verbose = 0;
    char *instrument = NULL;
    int c;
    while((c = getopt(argc, argv, "p:o:C:d:u:P:T::M:vh")) != EOF) {
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
            case 'P':
                instrument = optarg;
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
        case 0:  logging::init(instrument, LOG_INFO);     break;
        default: logging::init(instrument, LOG_DEBUG);    break;
    }
    logging::info("logging configured");
    if (!instrument) {
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

    std::string pvCfgFile;
    if (optind < argc)
        pvCfgFile = argv[optind];
    else {
        logging::critical("A PV config filename is mandatory");
        exit(1);
    }

    para.maxTrSize = 256 * 1024;

    Py_Initialize(); // for use by configuration
    Drp::EpicsArchApp app(para, pvCfgFile);
    app.run();
    app.handleReset(json({}));
    Py_Finalize(); // for use by configuration
}
