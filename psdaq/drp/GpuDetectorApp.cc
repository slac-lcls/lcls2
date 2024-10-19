#include "GpuDetectorApp.hh"

#include <getopt.h>
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/kwargs.hh"

#define PY_RELEASE_GIL    PyEval_SaveThread()
#define PY_ACQUIRE_GIL(x) PyEval_RestoreThread(x)
#define PY_RELEASE_GIL_GUARD    }
#define PY_ACQUIRE_GIL_GUARD(x) { PyGilGuard pyGilGuard(x);

using json = nlohmann::json;
using logging = psalg::SysLog;


namespace Drp
{

// Release GIL on exceptions, too
class PyGilGuard
{
public:
    PyGilGuard(PyThreadState*& pySave) : m_pySave(pySave)
    {
        PY_ACQUIRE_GIL(m_pySave);
    }
    ~PyGilGuard()
    {
        m_pySave = PY_RELEASE_GIL;
    }
private:
    PyThreadState*& m_pySave;
};


GpuDetectorApp::GpuDetectorApp(Parameters& para) :
    CollectionApp(para.collectionHost, para.partition, "drp", para.alias),
    m_drp(para, context()),
    m_para(para),
    m_det(nullptr),
    m_unconfigure(false)
{
    logging::info("GpuDetectorApp constructed in process ID %lu", syscall(SYS_gettid));

    Py_Initialize(); // for use by configuration
    m_pysave = PY_RELEASE_GIL; // Py_BEGIN_ALLOW_THREADS
}

// This initialization is in its own method (to be called from a higher layer)
// so that the dtor will run if it throws an exception.  This is needed to
// ensure Py_Finalize is executed.
void GpuDetectorApp::initialize()
{
    PY_ACQUIRE_GIL_GUARD(m_pysave);  // Py_END_ALLOW_THREADS

    // Register Detector types and the .so library that provides them
    auto& f = m_factory;   // Factory must remain in scope to avoid .so closing
    f.register_type("fakecam",   "libAreaDetector_gpu.so");
    //f.register_type<AreaDetectorGpu>("fakecam",   "libAreaDetector_gpu.so");
    //f.register_type<EpixHRemuGpu>   ("epixhremu", "libEpixHRemu_gpu.so");
    //f.register_type<EpixM320Gpu>    ("epixm320",  "libEpixM320_gpu.so");

    logging::info("m_gpu created in process ID %lu", syscall(SYS_gettid));

    m_gpu = f.create(m_para.detType, m_para, m_drp.pool);
    if (m_gpu == nullptr) {
        logging::critical("Error !! Could not create Detector object for %s", m_para.detType.c_str());
        throw "Could not create Detector object for " + m_para.detType;
    }
    printf("*** GpuDetectorApp::init: m_det %p\n", m_gpu->detector());
    m_det = m_gpu->detector();

    logging::info("Ready for transitions");

    PY_RELEASE_GIL_GUARD; // Py_BEGIN_ALLOW_THREADS
}

GpuDetectorApp::~GpuDetectorApp()
{
    printf("*** ~GpuDetectorApp()\n");

    // Try to take things down gracefully when an exception takes us off the
    // normal path so that the most chance is given for prints to show up
    handleReset(json({}));

    if (m_gpu)  delete m_gpu;

    try {
        PyGILState_Ensure();
        Py_Finalize(); // for use by configuration
    } catch(const std::exception &e) {
        std::cout << "Exception in ~GpuDetectorApp(): " << e.what() << "\n";
    } catch(...) {
        std::cout << "Exception in Python code: UNKNOWN\n";
    }

    printf("*** ~GpuDetectorApp() done\n");
}

void GpuDetectorApp::_disconnect()
{
    m_drp.disconnect();
    if (m_det)
        m_det->shutdown();
}

void GpuDetectorApp::_unconfigure()
{
    m_drp.pool.shutdown();              // Release Tr buffer pool
    if (m_gpuDetector) {
        m_gpuDetector->shutdown();
        if (m_exporter)  m_exporter.reset();
        if (m_gpuThread.joinable()) {
            m_gpuThread.join();
            logging::info("GpuReader thread finished");
        }
        if (m_collectorThread.joinable()) {
            m_collectorThread.join();
            logging::info("Collector thread finished");
        }
        m_gpuDetector.reset();
    }
    m_drp.unconfigure();

    if (m_det)
        m_det->namesLookup().clear();   // erase all elements

    m_unconfigure = false;
}

void GpuDetectorApp::handleConnect(const json& msg)
{
    json body = json({});

    PY_ACQUIRE_GIL_GUARD(m_pysave);  // Py_END_ALLOW_THREADS

    std::string errorMsg = m_drp.connect(msg, getId());
    if (!errorMsg.empty()) {
        logging::error("Error in DrpBase::connect()");
        logging::error("%s", errorMsg.c_str());
        body["err_info"] = errorMsg;
    }
    else {
        m_det->nodeId = m_drp.nodeId();
        m_det->connect(msg, std::to_string(getId()));
    }

    PY_RELEASE_GIL_GUARD; // Py_BEGIN_ALLOW_THREADS

    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void GpuDetectorApp::handleDisconnect(const json& msg)
{
    PY_ACQUIRE_GIL_GUARD(m_pysave);  // Py_END_ALLOW_THREADS

    // Carry out the queued Unconfigure, if there was one
    if (m_unconfigure) {
        _unconfigure();
    }

    _disconnect();

    PY_RELEASE_GIL_GUARD; // Py_BEGIN_ALLOW_THREADS

    json body = json({});
    reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void GpuDetectorApp::handlePhase1(const json& msg)
{
    printf("*** GpuDetectorApp::handlePhase1: m_det %p, %p\n", m_det, m_gpu->detector());

    std::string key = msg["header"]["key"];
    logging::debug("handlePhase1 for %s in GpuDetectorApp (m_det->scanEnabled() is %s)",
                   key.c_str(), m_det->scanEnabled() ? "TRUE" : "FALSE");

    json body = json({});

    PY_ACQUIRE_GIL_GUARD(m_pysave);  // Py_END_ALLOW_THREADS

    XtcData::Xtc& xtc = m_det->transitionXtc();
    xtc = {{XtcData::TypeId::Parent, 0}, {m_det->nodeId}};
    auto bufEnd = m_det->trXtcBufEnd();

    // @todo:
    //bool has_names_block_hex = false;
    //bool has_shapes_data_block_hex = false;
    //
    json phase1Info{ "" };
    // @todo:
    //if (msg.find("body") != msg.end()) {
    //    if (msg["body"].find("phase1Info") != msg["body"].end()) {
    //        phase1Info = msg["body"]["phase1Info"];
    //        if (phase1Info.find("NamesBlockHex") != phase1Info.end()) {
    //            has_names_block_hex = true;
    //        }
    //        if (phase1Info.find("ShapesDataBlockHex") != phase1Info.end()) {
    //            has_shapes_data_block_hex = true;
    //        }
    //    }
    //}

    if (key == "configure") {
        if (m_unconfigure) {
            _unconfigure();
        }
        // @todo:
        //if (has_names_block_hex && m_det->scanEnabled()) {
        //    std::string xtcHex = msg["body"]["phase1Info"]["NamesBlockHex"];
        //    unsigned hexlen = xtcHex.length();
        //    if (hexlen > 0) {
        //        logging::debug("configure phase1 in GpuDetectorApp: NamesBlockHex length=%u", hexlen);
        //        char *xtcBytes = new char[hexlen / 2]();
        //        if (_dehex(xtcHex, xtcBytes) != 0) {
        //            logging::error("configure phase1 in GpuDetectorApp: _dehex() failure");
        //        } else {
        //            logging::debug("configure phase1 in GpuDetectorApp: _dehex() success");
        //            // append the config xtc info to the dgram
        //            XtcData::Xtc& jsonxtc = *(XtcData::Xtc*)xtcBytes;
        //            logging::debug("configure phase1 jsonxtc.sizeofPayload() = %u\n",
        //                           jsonxtc.sizeofPayload());
        //            unsigned copylen = sizeof(XtcData::Xtc) + jsonxtc.sizeofPayload();
        //            auto payload = xtc.alloc(copylen, bufEnd);
        //            memcpy(payload, (const void*)xtcBytes, copylen);
        //        }
        //        delete[] xtcBytes;
        //    }
        //}

        std::string errorMsg = m_drp.configure(msg);
        if (!errorMsg.empty()) {
            errorMsg = "Phase 1 error: " + errorMsg;
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
        else {
            const std::string& config_alias = msg["body"]["config_alias"];
            printf("*** Make GpuDetector\n");
            m_gpuDetector = std::make_unique<GpuDetector>(m_para, m_drp, m_gpu);
            printf("*** Made GpuDetector\n");
            m_exporter = std::make_shared<Pds::MetricExporter>();
            if (m_drp.exposer()) {
                m_drp.exposer()->RegisterCollectable(m_exporter);
            }
            printf("*** Made Exporter\n");

            m_gpuThread = std::thread{&GpuDetector::reader, std::ref(*m_gpuDetector), m_exporter,
                                      std::ref(m_det), std::ref(m_drp.tebContributor())};
            printf("*** Started reader\n");
            m_collectorThread = std::thread(&GpuDetector::collector, std::ref(*m_gpuDetector),
                                            std::ref(m_drp.tebContributor()));
            printf("*** Started collector\n");

            // Provide EbReceiver with the Detector interface so that additional
            // data blocks can be formatted into the XTC, e.g. trigger information
            m_drp.ebReceiver().configure(m_det, m_gpuDetector.get());
            printf("*** Configured EbReceiver\n");

            // @todo: Maybe we should configure the h/w before the reader thread starts?
            unsigned error = m_det->configure(config_alias, xtc, bufEnd);
            printf("*** Configured Detector\n");
            if (!error) {
                // @todo:
                //json scan = _getscankeys(phase1Info, m_para.detName.c_str(), m_para.alias.c_str());
                //if (!scan.empty())
                //    error = m_det->configureScan(scan, xtc, bufEnd);
            }
            if (error) {
                std::string errorMsg = "Phase 1 error in Detector::configure()";
                body["err_info"] = errorMsg;
                logging::error("%s", errorMsg.c_str());
            }
            else {
                m_drp.runInfoSupport(xtc, bufEnd, m_det->namesLookup());
                m_drp.chunkInfoSupport(xtc, bufEnd, m_det->namesLookup());
            }
        }
    }
    else if (key == "unconfigure") {
        // "Queue" unconfiguration until after phase 2 has completed
        m_unconfigure = true;
    }
    else if (key == "beginstep") {
        // @todo:
        //// see if we find some step information in phase 1 that needs to be
        //// to be attached to the xtc
        //if (has_shapes_data_block_hex && m_det->scanEnabled()) {
        //    std::string xtcHex = msg["body"]["phase1Info"]["ShapesDataBlockHex"];
        //    unsigned hexlen = xtcHex.length();
        //    if (hexlen > 0) {
        //        logging::debug("beginstep phase1 in GpuDetectorApp: ShapesDataBlockHex length=%u", hexlen);
        //        char *xtcBytes = new char[hexlen / 2]();
        //        if (_dehex(xtcHex, xtcBytes) != 0) {
        //            logging::error("beginstep phase1 in GpuDetectorApp: _dehex() failure");
        //        } else {
        //            // append the beginstep xtc info to the dgram
        //            XtcData::Xtc& jsonxtc = *(XtcData::Xtc*)xtcBytes;
        //            logging::debug("beginstep phase1 jsonxtc.sizeofPayload() = %u\n",
        //                           jsonxtc.sizeofPayload());
        //            unsigned copylen = sizeof(XtcData::Xtc) + jsonxtc.sizeofPayload();
        //            auto payload = xtc.alloc(copylen, bufEnd);
        //            memcpy(payload, (const void*)xtcBytes, copylen);
        //        }
        //        delete[] xtcBytes;
        //    }
        //}

        unsigned error = m_det->beginstep(xtc, bufEnd, phase1Info);
        if (error) {
            logging::error("m_det->beginstep() returned error");
        } else {
            // @todo:
            //json scan = _getscanvalues(phase1Info, m_para.detName.c_str(), m_para.alias.c_str());
            //if (scan.empty()) {
            //    logging::debug("scan is empty");
            //} else {
            //    error = m_det->stepScan(scan, xtc, bufEnd);
            //    if (error) {
            //        logging::error("m_det->stepScan() returned error");
            //    }
            //}
        }
        if (error) {
            std::string errorMsg = "Phase 1 error in Detector::beginstep()";
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
    }
    else if (key == "beginrun") {
        RunInfo runInfo;
        std::string errorMsg = m_drp.beginrun(phase1Info, runInfo);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
        else {
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
    else if (key == "disable") {
        unsigned error = m_det->disable(xtc, bufEnd, phase1Info);
        if (error) {
            std::string errorMsg = "Phase 1 error in Detector::disable()";
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
        logging::debug("handlePhase1 disable complete");
    }

    PY_RELEASE_GIL_GUARD; // Py_BEGIN_ALLOW_THREADS

    json answer = createMsg(key, msg["header"]["msg_id"], getId(), body);
    reply(answer);

    logging::debug("handlePhase1 complete");
}

void GpuDetectorApp::handleReset(const json& msg)
{
    PY_ACQUIRE_GIL_GUARD(m_pysave);  // Py_END_ALLOW_THREADS

    unsubscribePartition();    // ZMQ_UNSUBSCRIBE
    _unconfigure();
    _disconnect();
    connectionShutdown();

    PY_RELEASE_GIL_GUARD; // Py_BEGIN_ALLOW_THREADS
}

json GpuDetectorApp::connectionInfo(const nlohmann::json& msg)
{
    std::string ip = m_para.kwargs.find("ep_domain") != m_para.kwargs.end()
                   ? getNicIp(m_para.kwargs["ep_domain"])
                   : getNicIp(m_para.kwargs["forceEnet"] == "yes");
    logging::debug("nic ip  %s", ip.c_str());
    json body = {{"connect_info", {{"nic_ip", ip}}}};

    PY_ACQUIRE_GIL_GUARD(m_pysave);  // Py_END_ALLOW_THREADS

    json info = m_det->connectionInfo(msg);
    body["connect_info"].update(info);
    json bufInfo = m_drp.connectionInfo(ip);
    body["connect_info"].update(bufInfo); // Revisit: Should be in det_info

    PY_RELEASE_GIL_GUARD; // Py_BEGIN_ALLOW_THREADS
    return body;
}

void GpuDetectorApp::connectionShutdown()
{
    m_drp.shutdown();
    if (m_exporter) {
        m_exporter.reset();
    }
}


}; // namespace Drp


int main(int argc, char* argv[])
{
    Drp::Parameters para;
    int c;
    std::string kwargs_str;
    std::string::size_type ii = 0;
    while((c = getopt(argc, argv, "p:o:l:D:S:C:d:u:k:P:M:v")) != EOF) {
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
            case 'D':
                para.detType = optarg;
                break;
            case 'S':
                para.serNo = optarg;
                break;
            case 'u':
                para.alias = optarg;
                break;
            case 'C':
                para.collectionHost = optarg;
                break;
            case 'd':
                para.device = optarg;
                break;
            case 'k':
                kwargs_str = kwargs_str.empty()
                           ? optarg
                           : kwargs_str + "," + optarg;
                break;
            case 'P':
                para.instrument = optarg;
                // remove station number suffix, if present
                ii = para.instrument.find(":");
                if (ii != std::string::npos) {
                    para.instrument.erase(ii, std::string::npos);
                }
                break;
            case 'M':
                para.prometheusDir = optarg;
                break;
            case 'v':
                ++para.verbose;
                break;
            default:
                return 1;
        }
    }

    switch (para.verbose) {
        case 0:  logging::init(para.instrument.c_str(), LOG_INFO);   break;
        default: logging::init(para.instrument.c_str(), LOG_DEBUG);  break;
    }
    logging::info("logging configured");
    if (optind < argc)
    {
        logging::error("Unrecognized argument:");
        while (optind < argc)
            logging::error("  %s ", argv[optind++]);
        return 1;
    }
    // Check required parameters
    if (para.instrument.empty()) {
        logging::warning("-P: instrument name is mandatory");
    }
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

    // Alias must be of form <detName>_<detSegment>
    size_t found = para.alias.rfind('_');
    if ((found == std::string::npos) || !isdigit(para.alias.back())) {
        logging::critical("-u: alias must have _N suffix");
        return 1;
    }
    para.detName = para.alias.substr(0, found);
    para.detSegment = std::stoi(para.alias.substr(found+1, para.alias.size()));

    get_kwargs(kwargs_str, para.kwargs);
    for (const auto& kwargs : para.kwargs)
    {
        if (kwargs.first == "forceEnet")      continue;
        if (kwargs.first == "ep_fabric")      continue;
        if (kwargs.first == "ep_domain")      continue;
        if (kwargs.first == "ep_provider")    continue;
        if (kwargs.first == "sim_length")     continue;  // XpmDetector
        if (kwargs.first == "timebase")       continue;  // XpmDetector
        if (kwargs.first == "pebbleBufSize")  continue;  // DrpBase
        if (kwargs.first == "pebbleBufCount") continue;  // DrpBase
        if (kwargs.first == "batching")       continue;  // DrpBase
        if (kwargs.first == "directIO")       continue;  // DrpBase
        if (kwargs.first == "pva_addr")       continue;  // DrpBase
        if (kwargs.first == "gpuId")          continue;
        logging::critical("Unrecognized kwarg '%s=%s'\n",
                          kwargs.first.c_str(), kwargs.second.c_str());
        return 1;
    }

    para.maxTrSize = 256 * 1024;
    try {
        Drp::GpuDetectorApp app(para);
        app.initialize();
        app.run();
        std::cout<<"end of drp main\n";
        return 0;
    }
    catch (std::exception& e)  { logging::critical("%s", e.what()); }
    catch (std::string& e)     { logging::critical("%s", e.c_str()); }
    catch (char const* e)      { logging::critical("%s", e); }
    catch (...)                { logging::critical("Default exception"); }
    return EXIT_FAILURE;
}
