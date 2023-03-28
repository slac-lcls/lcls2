#include <iostream>
#include <iomanip>
#include <string>
#include <future>
#include <thread>
#include "drp.hh"
#include "Detector.hh"
#include "TimingBEB.hh"
#include "TimingSystem.hh"
#include "TimeTool.hh"
#include "AreaDetector.hh"
#include "Digitizer.hh"
#include "EpixQuad.hh"
#include "EpixHR2x2.hh"
#include "Epix100.hh"
#include "Opal.hh"
#include "Wave8.hh"
#include "Piranha4.hh"
#include "psdaq/service/MetricExporter.hh"
#include "PGPDetectorApp.hh"
#include "psalg/utils/SysLog.hh"
#include "RunInfoDef.hh"
#include "psdaq/service/IpcUtils.hh"


#define PY_RELEASE_GIL    PyEval_SaveThread()
#define PY_ACQUIRE_GIL(x) PyEval_RestoreThread(x)
//#define PY_RELEASE_GIL    0
//#define PY_ACQUIRE_GIL(x) {}
#define PY_RELEASE_GIL_GUARD    }
#define PY_ACQUIRE_GIL_GUARD(x) { PyGilGuard pyGilGuard(x);

using json = nlohmann::json;
using logging = psalg::SysLog;
using std::string;
using namespace Pds::Ipc;

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
            if (delim != string::npos) {
                string src = v.substr(0,delim);
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
            if (delim != string::npos) {
                string src = it.key().substr(0,delim);
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

int startDrpPython(pid_t& pyPid, unsigned workerNum, unsigned keyBase, long shmemSize, const Parameters& para, DrpBase& drp)
{
    // Fork
    pyPid = fork();

    if (pyPid == pid_t(0))
    {
        time_t my_time = time(NULL);
    
        std::cout << "DEBUG: Thread " << workerNum << "%u]" << ctime(&my_time) << " - Thread num: %s" << std::this_thread::get_id() << std::endl;

        // Executing external code 
        execlp("python",
               "python",
               "-u",
               "-m",
               "psdaq.drp.drp_python",
               std::to_string(keyBase).c_str(),
               std::to_string(drp.pool.pebble.bufferSize()).c_str(),
               std::to_string(para.maxTrSize).c_str(),
               std::to_string(shmemSize).c_str(),
               para.detName.c_str(),
               std::to_string(para.detSegment).c_str(),
               std::to_string(workerNum).c_str(),
               nullptr);

        // Execlp returns only on error                    
        logging::critical("Error on 'execlp python' for worker %d ': %m", workerNum);
        abort();
    } else {
        return 0;
    }
}

void PGPDetectorApp::setupDrpPython() {
    const unsigned KEY_BASE = 40000;

    size_t shmemSize = m_drp.pool.pebble.bufferSize();
    if (m_para.maxTrSize > shmemSize) shmemSize=m_para.maxTrSize;

    // Round up to an integral number of pages
    long pageSize = sysconf(_SC_PAGESIZE);
    shmemSize = (shmemSize + pageSize - 1) & ~(pageSize - 1);

    std::vector<std::thread> drpPythonThreads;

    for (unsigned workerNum=0; workerNum<m_para.nworkers; workerNum++) {

        unsigned keyBase  =  KEY_BASE + 1000 * workerNum + 100 * m_para.partition;

        // Creating message queues
        int rc = setupDrpMsgQueue(keyBase+0, "Inputs", m_inpMqId[workerNum], workerNum);
        if (rc) {
            cleanupDrpPython(m_inpMqId, m_resMqId, m_inpShmId, m_resShmId, m_para.nworkers);
            logging::critical("[Thread %u] error setting up Drp message queues", workerNum);
            abort();
        }
        rc = setupDrpMsgQueue(keyBase+1, "Results", m_resMqId[workerNum], workerNum);
        if (rc) {
            cleanupDrpPython(m_inpMqId, m_resMqId, m_inpShmId, m_resShmId, m_para.nworkers);
            logging::critical("[Thread %u] error setting up Drp message queues", workerNum);
            abort();
        }

        // Creating shared memory
        size_t shmemSize = m_drp.pool.pebble.bufferSize();
        if (m_para.maxTrSize > shmemSize) shmemSize=m_para.maxTrSize;
        
        // Round up to an integral number of pages
        long pageSize = sysconf(_SC_PAGESIZE);
        shmemSize = (shmemSize + pageSize - 1) & ~(pageSize - 1);

        rc = setupDrpShMem(keyBase+2, shmemSize, "Inputs", m_inpShmId[workerNum], workerNum);
        if (rc) {
            cleanupDrpPython(m_inpMqId, m_resMqId, m_inpShmId, m_resShmId, m_para.nworkers);
            logging::critical("[Thread %u] error setting up Drp shared memory buffers", workerNum);
            abort();
        }

        rc = setupDrpShMem(keyBase+3, shmemSize, "Results", m_resShmId[workerNum], workerNum);
        if (rc) {
            cleanupDrpPython(m_inpMqId, m_resMqId, m_inpShmId, m_resShmId, m_para.nworkers);
            logging::critical("[Thread %u] error setting up Drp shared memory buffers", workerNum);
            abort();
        }

        logging::info("IPC set up for worker %d", workerNum);

        drpPythonThreads.emplace_back(startDrpPython,
                                      std::ref(m_drpPids[workerNum]),
                                      workerNum,
                                      keyBase,
                                      shmemSize,
                                      std::ref(m_para),
                                      std::ref(m_drp));
    }


    for (std::vector<std::thread>::iterator thrIter =drpPythonThreads.begin();
        thrIter != drpPythonThreads.end(); thrIter++) {
        if (thrIter->joinable()) {
            thrIter->join();
        }
    }     

    logging::info("Drp python processes started");
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
    m_drp(para, context()),
    m_para(para),
    m_det(nullptr),
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

    Factory<Detector> f;
    f.register_type<AreaDetector>("fakecam");
    f.register_type<AreaDetector>("cspad");
    f.register_type<Digitizer>   ("hsd");
    f.register_type<EpixQuad>    ("epixquad");
    f.register_type<EpixHR2x2>   ("epixhr2x2");
    f.register_type<Epix100>     ("epix100");
    f.register_type<Opal>        ("opal");
    f.register_type<TimeTool>    ("tt");
    f.register_type<TimingBEB>   ("tb");
    f.register_type<TimingSystem>("ts");
    f.register_type<Wave8>       ("wave8");
    f.register_type<Piranha4>    ("piranha4");


    m_det = f.create(&m_para, &m_drp.pool);
    if (m_det == nullptr) {
        logging::critical("Error !! Could not create Detector object for %s", m_para.detType.c_str());
        throw "Could not create Detector object for " + m_para.detType;
    }

    // Provide EbReceiver with the Detector interface so that additional
    // data blocks can be formatted into the XTC, e.g. trigger information
    m_drp.ebReceiver().detector(m_det);

    // Initialize these to zeros. They will store the file descriptors and
    // process numbers if Drp Python is used or be just zeros if it is not.
    m_inpMqId = new int[m_para.nworkers]();
    m_resMqId = new int[m_para.nworkers]();
    m_inpShmId = new int[m_para.nworkers]();
    m_resShmId = new int[m_para.nworkers]();
    m_drpPids = new pid_t[m_para.nworkers]();

    auto kwargs_it = m_para.kwargs.find("drp");
    if (kwargs_it != m_para.kwargs.end() && kwargs_it->second == "python") {
        logging::info("Starting DrpPython");
        setupDrpPython();
    }

    logging::info("Ready for transitions");

    PY_RELEASE_GIL_GUARD; // Py_BEGIN_ALLOW_THREADS
}

PGPDetectorApp::~PGPDetectorApp()
{
    // Try to take things down gracefully when an exception takes us off the
    // normal path so that the most chance is given for prints to show up
    handleReset(json({}));

    if (m_det)  delete m_det;

    try {
        PyGILState_Ensure();
        Py_Finalize(); // for use by configuration
    } catch(const std::exception &e) {
        std::cout << "Exception in ~PGPDetectorApp(): " << e.what() << "\n";
    } catch(...) {
        std::cout << "Exception in Python code: UNKNOWN\n";
    }
}

void PGPDetectorApp::disconnect()
{
    m_drp.disconnect();  
    if (m_det)
        m_det->shutdown();
}

void PGPDetectorApp::unconfigure()
{
    m_drp.pool.shutdown();              // Release Tr buffer pool
    if (m_pgpDetector) {
        m_pgpDetector->shutdown();
        if (m_pgpThread.joinable()) {
            m_pgpThread.join();
            logging::info("PGPReader thread finished");
        }
        if (m_collectorThread.joinable()) {
            m_collectorThread.join();
            logging::info("Collector thread finished");
        }
        m_exporter.reset();
        m_pgpDetector.reset();
    }
    m_drp.unconfigure();

    if (m_det)
        m_det->namesLookup().clear();   // erase all elements

    m_unconfigure = false;
}

void PGPDetectorApp::handleConnect(const json& msg)
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

void PGPDetectorApp::handleDisconnect(const json& msg)
{
    PY_ACQUIRE_GIL_GUARD(m_pysave);  // Py_END_ALLOW_THREADS

    // Carry out the queued Unconfigure, if there was one
    if (m_unconfigure) {
        unconfigure();
    }

    disconnect();

    PY_RELEASE_GIL_GUARD; // Py_BEGIN_ALLOW_THREADS

    json body = json({});
    reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void PGPDetectorApp::handlePhase1(const json& msg)
{
    std::string key = msg["header"]["key"];
    logging::debug("handlePhase1 for %s in PGPDetectorApp (m_det->scanEnabled() is %s)",
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
        if (m_unconfigure) {
            unconfigure();
        }
        if (has_names_block_hex && m_det->scanEnabled()) {
            std::string xtcHex = msg["body"]["phase1Info"]["NamesBlockHex"];
            unsigned hexlen = xtcHex.length();
            if (hexlen > 0) {
                logging::debug("configure phase1 in PGPDetectorApp: NamesBlockHex length=%u", hexlen);
                char *xtcBytes = new char[hexlen / 2]();
                if (_dehex(xtcHex, xtcBytes) != 0) {
                    logging::error("configure phase1 in PGPDetectorApp: _dehex() failure");
                } else {
                    logging::debug("configure phase1 in PGPDetectorApp: _dehex() success");
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

        std::string errorMsg = m_drp.configure(msg);
        if (!errorMsg.empty()) {
            errorMsg = "Phase 1 error: " + errorMsg;
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
        else {
            m_pgpDetector = std::make_unique<PGPDetector>(m_para, m_drp, m_det, m_inpMqId, m_resMqId, m_inpShmId, m_resShmId);
            m_exporter = std::make_shared<Pds::MetricExporter>();
            if (m_drp.exposer()) {
                m_drp.exposer()->RegisterCollectable(m_exporter);
            }

            m_pgpThread = std::thread{&PGPDetector::reader, std::ref(*m_pgpDetector), m_exporter,
                                      std::ref(m_det), std::ref(m_drp.tebContributor())};
            m_collectorThread = std::thread(&PGPDetector::collector, std::ref(*m_pgpDetector),
                                            std::ref(m_drp.tebContributor()));

            std::string config_alias = msg["body"]["config_alias"];
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
        // see if we find some step information in phase 1 that needs to be
        // to be attached to the xtc
        if (has_shapes_data_block_hex && m_det->scanEnabled()) {
            std::string xtcHex = msg["body"]["phase1Info"]["ShapesDataBlockHex"];
            unsigned hexlen = xtcHex.length();
            if (hexlen > 0) {
                logging::debug("beginstep phase1 in PGPDetectorApp: ShapesDataBlockHex length=%u", hexlen);
                char *xtcBytes = new char[hexlen / 2]();
                if (_dehex(xtcHex, xtcBytes) != 0) {
                    logging::error("beginstep phase1 in PGPDetectorApp: _dehex() failure");
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

void PGPDetectorApp::handleReset(const json& msg)
{
    PY_ACQUIRE_GIL_GUARD(m_pysave);  // Py_END_ALLOW_THREADS

    unsubscribePartition();    // ZMQ_UNSUBSCRIBE
    unconfigure();
    disconnect();
    connectionShutdown();

    PY_RELEASE_GIL_GUARD; // Py_BEGIN_ALLOW_THREADS
}

json PGPDetectorApp::connectionInfo()
{
    std::string ip = m_para.kwargs.find("ep_domain") != m_para.kwargs.end()
                   ? getNicIp(m_para.kwargs["ep_domain"])
                   : getNicIp(m_para.kwargs["forceEnet"] == "yes");
    logging::debug("nic ip  %s", ip.c_str());
    json body = {{"connect_info", {{"nic_ip", ip}}}};
    json info = m_det->connectionInfo();
    body["connect_info"].update(info);
    json bufInfo = m_drp.connectionInfo(ip);
    body["connect_info"].update(bufInfo); // Revisit: Should be in det_info
    return body;
}

void PGPDetectorApp::connectionShutdown()
{
    m_drp.shutdown();
    if (m_exporter) {
        m_exporter.reset();
    }
}
}
