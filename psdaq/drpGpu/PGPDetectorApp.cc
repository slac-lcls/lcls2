#include "PGPDetectorApp.hh"

#include "PGPDetector.hh"

#include <getopt.h>
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/kwargs.hh"
#include "xtcdata/xtc/TransitionId.hh"

#ifndef NVTX_DISABLE                    // Defined, or not, in drpGpu/MemPool.hh
#include <cuda_profiler_api.h>
#endif

#define PY_RELEASE_GIL    PyEval_SaveThread()
#define PY_ACQUIRE_GIL(x) PyEval_RestoreThread(x)
#define PY_RELEASE_GIL_GUARD    }
#define PY_ACQUIRE_GIL_GUARD(x) { PyGilGuard pyGilGuard(x);

using json = nlohmann::json;
using logging = psalg::SysLog;
using namespace XtcData;

static const char* const MAG_ON  = "\033[0;35m";
static const char* const COL_OFF = "\033[0m";


// _dehex - convert a hex std::string to an array of chars
//
// For example, string "0E2021" is converted to array [14, 32, 33].
// <outArray> must be allocated by the caller to at least half
// the length of <inString>.
//
// RETURNS: 0 on success, otherwise 1.
//
static int _dehex(std::string inString, char *outArray)
{
    if (outArray) {
        try {
            for (unsigned uu = 0; uu < inString.length() / 2; uu++) {
                std::string str2 = inString.substr(2*uu, 2);
                outArray[uu] = (char) std::stoi(str2, 0, 16);   // base 16
            }
            return 0;   // success
        }
        catch (std::exception& e) {
            std::cout << "Exception in _dehex(): " << e.what() << "\n";
        }
    }
    return 1;           // error
}

//  Return a list of scan parameters for detname
static json _getscankeys(const json& stepInfo, const char* detname, const char* alias)
{
    json update;
//  bool detscanned=false;
    if (stepInfo.contains("step_keys")) {
        json reconfig = stepInfo["step_keys"];
        logging::debug("_getscankeys reconfig [%s]",reconfig.dump().c_str());
        for (json::iterator it=reconfig.begin(); it != reconfig.end(); it++) {
            std::string v = it->get<std::string>();
            logging::debug("_getscankeys key [%s]",v.c_str());
            size_t delim = v.find(":");
            if (delim != std::string::npos) {
                std::string src = v.substr(0,delim);
                if (src == alias)
                    update.push_back(v.substr(delim+1));
//              if (src.substr(0,src.rfind("_",delim)) == detname)
//                  detscanned = true;
            }
        }
    }

    logging::debug("_getscankeys returning [%s]",update.dump().c_str());
    return update;
}

//  Return a dictionary of scan parameters for detname
static json _getscanvalues(const json& stepInfo, const char* detname, const char* alias)
{
    json update;
//  bool detscanned=false;
    if (stepInfo.contains("step_values")) {
        json reconfig = stepInfo["step_values"];
        for (json::iterator it=reconfig.begin(); it != reconfig.end(); it++) {
            std::string v = it.key();
            logging::debug("_getscanvalues key [%s]",v.c_str());
            size_t delim = it.key().find(":");
            if (delim != std::string::npos) {
                std::string src = it.key().substr(0,delim);
                if (src == alias)
                    update[it.key().substr(delim+1)] = it.value();
//              if (src.substr(0,src.rfind("_",delim)) == detname)
//                  detscanned = true;
            }
        }
    }

    logging::debug("_getscanvalues returning [%s]",update.dump().c_str());
    return update;
}

namespace Drp {
  namespace Gpu {

void DetectorFactory::register_type(const std::string& name, const std::string& solib)
{
    // This makes a copy of the solib string in the map so it's available for create()
    m_create_funcs.emplace(name, solib);
}

Drp::Gpu::Detector* DetectorFactory::create(const std::string& name,
                                            Parameters&        para,
                                            MemPoolGpu&        pool)
{
    auto it = m_create_funcs.find(name);
    if (it == m_create_funcs.end())  return nullptr;

    return _instantiate(m_dl, it->second, para, pool);
}

Drp::Gpu::Detector* DetectorFactory::_instantiate(Pds::Dl&           dl,
                                                  const std::string& soName,
                                                  Parameters&        para,
                                                  MemPoolGpu&        pool)
{
    logging::debug("Loading library '%s'", soName.c_str());

    if (dl.open(soName, RTLD_LAZY))
    {
        logging::error("Error opening library '%s'", soName.c_str());
        return nullptr;
    }

    const std::string symName("createDetector");
    auto createFn = dl.loadSymbol(symName.c_str());
    if (!createFn)
    {
        logging::error("Symbol '%s' not found in %s", symName.c_str(), soName.c_str());
        return nullptr;
    }
    typedef Drp::Gpu::Detector* fn_t(Parameters& para, MemPool& pool);
    auto instance = reinterpret_cast<fn_t*>(createFn)(para, pool);
    if (!instance)
    {
        logging::error("%Error calling %s from %s", symName.c_str(), soName.c_str());
        return nullptr;
    }
    return instance;
}


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


PGPDetectorApp::PGPDetectorApp(Parameters& para) :
    CollectionApp(para.collectionHost, para.partition, "drp", para.alias),
    m_para       (para),
    m_pool       (para),
    m_det        (nullptr),
    m_unconfigure(false)
{
    Py_Initialize(); // for use by configuration
    m_pysave = PY_RELEASE_GIL; // Py_BEGIN_ALLOW_THREADS
}

// This initialization is in its own method (to be called from a higher layer)
// so that the dtor will run if it throws an exception.  This is needed to
// ensure Py_Finalize is executed.
void PGPDetectorApp::initialize()
{
    PY_ACQUIRE_GIL_GUARD(m_pysave);  // Py_END_ALLOW_THREADS

    // Register Detector types and the .so library that provides them
    auto& f = m_factory;   // Factory must remain in scope to avoid .so closing
    f.register_type("fakecam",   "libAreaDetector_gpu.so");
    f.register_type("epixuhremu", "libEpixUHRemu_gpu.so");
    f.register_type("epixuhrsim", "libEpixUHRsim_gpu.so");
    //f.register_type("epixuhr",   "libEpixUHR_gpu.so");

    m_det = f.create(m_para.detType, m_para, m_pool);
    if (m_det == nullptr) {
        logging::critical("Could not create GPU Detector object for '%s'", m_para.detType.c_str());
        throw "Could not create Detector object for " + m_para.detType;
    }

    m_drp = std::make_unique<PGPDrp>(m_para, m_pool, *m_det, context());

    logging::info("Ready for transitions");

    PY_RELEASE_GIL_GUARD; // Py_BEGIN_ALLOW_THREADS
}

PGPDetectorApp::~PGPDetectorApp()
{
    logging::debug("PGPDetectorApp::dtor");

    // Try to take things down gracefully when an exception takes us off the
    // normal path so that the most chance is given for prints to show up
    handleReset(json({}));

    if (m_det)  delete m_det;

    try {
        PyGILState_Ensure();
        Py_Finalize(); // for use by configuration
    } catch(const std::exception &e) {
        std::cout << "Exception in ~Gpu::PGPDetectorApp(): " << e.what() << "\n";
    } catch(...) {
        std::cout << "Exception in Python code: UNKNOWN\n";
    }
}

void PGPDetectorApp::_disconnect()
{
    logging::debug("PGPDetectorApp::_disconnect");

    m_drp->disconnect();
    if (m_det)
        m_det->shutdown();
}

void PGPDetectorApp::_unconfigure()
{
    logging::debug("PGPDetectorApp::_unconfigure");

    m_drp->pool.shutdown();              // Release Tr buffer pool
    m_drp->unconfigure();

    if (m_det)
        m_det->namesLookup().clear();   // erase all elements

    m_unconfigure = false;
}

void PGPDetectorApp::handleConnect(const json& msg)
{
    logging::debug("PGPDetectorApp::handleConnect");

    json body = json({});

    PY_ACQUIRE_GIL_GUARD(m_pysave);  // Py_END_ALLOW_THREADS

    std::string errorMsg = m_drp->connect(msg, getId());
    if (!errorMsg.empty()) {
        logging::error("Error in DrpBase::connect()");
        logging::error("%s", errorMsg.c_str());
        body["err_info"] = errorMsg;
    }
    else {
        m_det->nodeId = m_drp->nodeId();
        m_det->connect(msg, std::to_string(getId()));
    }

    PY_RELEASE_GIL_GUARD; // Py_BEGIN_ALLOW_THREADS

    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void PGPDetectorApp::handleDisconnect(const json& msg)
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

void PGPDetectorApp::handlePhase1(const json& msg)
{
    std::string key = msg["header"]["key"];
    TransitionId::Value tid;
    logging::debug("handlePhase1 for %s in Gpu::PGPDetectorApp (m_det->scanEnabled() is %s)",
                   key.c_str(), m_det->scanEnabled() ? "TRUE" : "FALSE");

    json body = json({});

    PY_ACQUIRE_GIL_GUARD(m_pysave);  // Py_END_ALLOW_THREADS

    XtcData::Xtc& xtc = m_det->transitionXtc();
    xtc = {{XtcData::TypeId::Parent, 0}, {m_det->nodeId}};
    auto bufEnd = m_det->trXtcBufEnd();

    bool has_names_block_hex = false;
    bool has_shapes_data_block_hex = false;

    json phase1Info{ "" };
    if (msg.find("body") != msg.end()) {
        if (msg["body"].find("phase1Info") != msg["body"].end()) {
            phase1Info = msg["body"]["phase1Info"];
            if (phase1Info.find("NamesBlockHex") != phase1Info.end()) {
                has_names_block_hex = true;
            }
            if (phase1Info.find("ShapesDataBlockHex") != phase1Info.end()) {
                has_shapes_data_block_hex = true;
            }
        }
    }

    if (key == "configure") {
        tid = TransitionId::Configure;
        if (m_unconfigure) {
            _unconfigure();
        }
        if (has_names_block_hex && m_det->scanEnabled()) {
           std::string xtcHex = msg["body"]["phase1Info"]["NamesBlockHex"];
           unsigned hexlen = xtcHex.length();
           if (hexlen > 0) {
               logging::debug("configure phase1 in Gpu::PGPDetectorApp: NamesBlockHex length=%u", hexlen);
               char *xtcBytes = new char[hexlen / 2]();
               if (_dehex(xtcHex, xtcBytes) != 0) {
                   logging::error("configure phase1 in Gpu::PGPDetectorApp: _dehex() failure");
               } else {
                   logging::debug("configure phase1 in Gpu::PGPDetectorApp: _dehex() success");
                   // append the config xtc info to the dgram
                   XtcData::Xtc& jsonxtc = *(XtcData::Xtc*)xtcBytes;
                   logging::debug("configure phase1 jsonxtc.sizeofPayload() = %u\n",
                                  jsonxtc.sizeofPayload());
                   unsigned copylen = sizeof(XtcData::Xtc) + jsonxtc.sizeofPayload();
                   auto payload = xtc.alloc(copylen, bufEnd);
                   memcpy(payload, (const void*)xtcBytes, copylen);
               }
               delete[] xtcBytes;
           }
        }

        // Configure the detector first
        const std::string& config_alias = msg["body"]["config_alias"];
        unsigned error = m_det->configure(config_alias, xtc, bufEnd);
        if (!error) {
            json scan = _getscankeys(phase1Info, m_para.detName.c_str(), m_para.alias.c_str());
            if (!scan.empty())
                error = m_det->configureScan(scan, xtc, bufEnd);
        }
        if (error) {
            std::string errorMsg = "Phase 1 error in Detector::configure()";
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
        else {
            // Next, configure the DRP
            std::string errorMsg = m_drp->configure(msg);
            if (!errorMsg.empty()) {
                errorMsg = "Phase 1 error: " + errorMsg;
                body["err_info"] = errorMsg;
                logging::error("%s", errorMsg.c_str());
            }
            else {
                m_drp->runInfoSupport(xtc, bufEnd, m_det->namesLookup());
                m_drp->chunkInfoSupport(xtc, bufEnd, m_det->namesLookup());
                m_drp->reducerConfigure(xtc, bufEnd);
            }
        }
    }
    else if (key == "unconfigure") {
        tid = TransitionId::Unconfigure;
        // "Queue" unconfiguration until after phase 2 has completed
        m_unconfigure = true;
    }
    else if (key == "beginstep") {
        tid = TransitionId::BeginStep;
        // see if we find some step information in phase 1 that needs to be
        // to be attached to the xtc
        if (has_shapes_data_block_hex && m_det->scanEnabled()) {
            std::string xtcHex = msg["body"]["phase1Info"]["ShapesDataBlockHex"];
            unsigned hexlen = xtcHex.length();
            if (hexlen > 0) {
                logging::debug("beginstep phase1 in Gpu::PGPDetectorApp: ShapesDataBlockHex length=%u", hexlen);
                char *xtcBytes = new char[hexlen / 2]();
                if (_dehex(xtcHex, xtcBytes) != 0) {
                    logging::error("beginstep phase1 in Gpu::PGPDetectorApp: _dehex() failure");
                } else {
                    // append the beginstep xtc info to the dgram
                    XtcData::Xtc& jsonxtc = *(XtcData::Xtc*)xtcBytes;
                    logging::debug("beginstep phase1 jsonxtc.sizeofPayload() = %u\n",
                                   jsonxtc.sizeofPayload());
                    unsigned copylen = sizeof(XtcData::Xtc) + jsonxtc.sizeofPayload();
                    auto payload = xtc.alloc(copylen, bufEnd);
                    memcpy(payload, (const void*)xtcBytes, copylen);
                }
                delete[] xtcBytes;
            }
        }

        unsigned error = m_det->beginstep(xtc, bufEnd, phase1Info);
        if (error) {
            logging::error("m_det->beginstep() returned error");
        } else {
            json scan = _getscanvalues(phase1Info, m_para.detName.c_str(), m_para.alias.c_str());
            if (scan.empty()) {
                logging::debug("scan is empty");
            } else {
                error = m_det->stepScan(scan, xtc, bufEnd);
                if (error) {
                    logging::error("m_det->stepScan() returned error");
                }
            }
        }
        if (error) {
            std::string errorMsg = "Phase 1 error in Detector::beginstep()";
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
    }
    else if (key == "endstep") {
        tid = TransitionId::EndStep;
    }
    else if (key == "beginrun") {
        tid = TransitionId::BeginRun;
        RunInfo runInfo;
        std::string errorMsg = m_drp->beginrun(phase1Info, runInfo);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
        else {
            m_drp->runInfoData(xtc, bufEnd, m_det->namesLookup(), runInfo);
        }
    }
    else if (key == "endrun") {
        tid = TransitionId::EndRun;
        std::string errorMsg = m_drp->endrun(phase1Info);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
    }
    else if (key == "enable") {
        tid = TransitionId::Enable;
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
#ifndef NVTX_DISABLE
        logging::info("%sEnabling GPU Profiler data collection%s", MAG_ON, COL_OFF);
        cudaProfilerStart();
#endif
        logging::debug("handlePhase1 enable complete");
    }
    else if (key == "disable") {
        tid = TransitionId::Disable;
#ifndef NVTX_DISABLE
        cudaProfilerStop();
        logging::info("%sDisabled GPU Profiler data collection%s", MAG_ON, COL_OFF);
#endif
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

    // Trigger phase 2 if we're in simulator mode
    if (m_para.device == "/dev/null") { // Simulator mode
      auto det = m_det->gpuDetector();
      if (det)  det->issuePhase2(tid);
    }
}

void PGPDetectorApp::handleReset(const json& msg)
{
    logging::debug("PGPDetectorApp::handleReset");

    PY_ACQUIRE_GIL_GUARD(m_pysave);  // Py_END_ALLOW_THREADS

    unsubscribePartition();    // ZMQ_UNSUBSCRIBE
    _unconfigure();
    _disconnect();
    connectionShutdown();

    PY_RELEASE_GIL_GUARD; // Py_BEGIN_ALLOW_THREADS
}

void PGPDetectorApp::handleDealloc(const json& msg)
{
    PY_ACQUIRE_GIL_GUARD(m_pysave);  // Py_END_ALLOW_THREADS
    CollectionApp::handleDealloc(msg);
    PY_RELEASE_GIL_GUARD; // Py_BEGIN_ALLOW_THREADS
}

json PGPDetectorApp::connectionInfo(const nlohmann::json& msg)
{
    logging::debug("PGPDetectorApp::connectionInfo");

    std::string ip = m_para.kwargs.find("ep_domain") != m_para.kwargs.end()
                   ? getNicIp(m_para.kwargs["ep_domain"])
                   : getNicIp(m_para.kwargs["forceEnet"] == "yes");
    logging::debug("nic ip  %s", ip.c_str());
    json body = {{"connect_info", {{"nic_ip", ip}}}};

    PY_ACQUIRE_GIL_GUARD(m_pysave);  // Py_END_ALLOW_THREADS

  printf("*** PGPDetectorApp::connectionInfo 1\n");
    json info = m_det->connectionInfo(msg);
  printf("*** PGPDetectorApp::connectionInfo 1a\n");
    body["connect_info"].update(info);
  printf("*** PGPDetectorApp::connectionInfo 2\n");
    json bufInfo = m_drp->connectionInfo(ip);
    body["connect_info"].update(bufInfo); // Revisit: Should be in det_info
  printf("*** PGPDetectorApp::connectionInfo 3\n");

    PY_RELEASE_GIL_GUARD; // Py_BEGIN_ALLOW_THREADS

    return body;
}

void PGPDetectorApp::connectionShutdown()
{
    logging::debug("PGPDetectorApp::connectionShutdown");

    if (m_det) {
        m_det->connectionShutdown();
    }

    m_drp->DrpBase::shutdown();
}

  } // Gpu
} // Drp


int main(int argc, char* argv[])
{
    Drp::Parameters para;
    int c;
    std::string kwargs_str;
    std::string::size_type ii = 0;
    while((c = getopt(argc, argv, "p:o:l:D:S:C:d:u:k:P:M:W:v")) != EOF) {
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
            case 'W':
                para.nworkers = std::stoi(optarg);
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
        if (kwargs.first == "dmaSize")        continue;  // GPU DRP
        if (kwargs.first == "gpuId")          continue;  // GPU DRP
        if (kwargs.first == "reducer")        continue;  // GPU DRP
        if (kwargs.first == "sim_l1_delay")   continue;  // GPU DRP Simulator
        if (kwargs.first == "sim_su_rate")    continue;  // GPU DRP Simulator
        logging::critical("Unrecognized kwarg '%s=%s'\n",
                          kwargs.first.c_str(), kwargs.second.c_str());
        return 1;
    }

    // Set up signal handler
    initShutdownSignals(para.alias, [](){ exit(0); });

    para.batchSize = 1; // Max # of DMA buffers queued for freeing - Must be a power of 2
    para.maxTrSize = 256 * 1024;
    try {
        Drp::Gpu::PGPDetectorApp app(para);
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
