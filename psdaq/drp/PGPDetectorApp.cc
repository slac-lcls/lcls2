#include <iostream>
#include <iomanip>
#include <string>
#include <future>
#include <thread>
#include <cstdio>
#include "drp.hh"
#include "Detector.hh"
#include "TimingBEB.hh"
#include "TimingSystem.hh"
#include "TimeTool.hh"
#include "AreaDetector.hh"
#include "Digitizer.hh"
#include "EpixQuad.hh"
#include "EpixHR2x2.hh"
#include "EpixHRemu.hh"
#include "EpixM320.hh"
#include "EpixUHR.hh"
#include "Epix100.hh"
#include "JungfrauEmulator.hh"
#include "Opal.hh"
#include "HREncoder.hh"
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

static int cleanupDrpPython(std::string keyBase, int* inpMqId, int* resMqId, int* inpShmId, int* resShmId,
                            unsigned numWorkers)
{
    int rcSave = 0;
    for (unsigned workerNum=0; workerNum<numWorkers; workerNum++) {
        int rc;
        if (inpShmId[workerNum]) {
            rc = cleanupDrpShmMem("/shminp_" + keyBase + "_" + std::to_string(workerNum), inpShmId[workerNum]);
            if (rc) {
                logging::error("Error cleaning up Inputs Shared Memory for worker %d: %m", workerNum);
                rcSave = rc;
            }
            inpShmId[workerNum] = 0;
        }
        if (resShmId[workerNum]) {
            rc = cleanupDrpShmMem("/shmres_" + keyBase + "_" + std::to_string(workerNum), resShmId[workerNum]);
            if (rc) {
                logging::error("Error cleaning up Results Shared Memory for worker %d: %m", workerNum);
                rcSave = rc;
            }
            resShmId[workerNum] = 0;
        }

        if (inpMqId[workerNum]) {
            rc = cleanupDrpMq("/mqinp_" + keyBase  + "_" + std::to_string(workerNum), inpMqId[workerNum]);
            if (rc) {
                logging::error("Error cleaning up Inputs Message Queue for worker %d: %m", workerNum);
                rcSave = rc;
            }
            inpMqId[workerNum] = 0;
        }

        if (resMqId[workerNum]) {
            rc = cleanupDrpMq("/mqres_" + keyBase  + "_" + std::to_string(workerNum), resMqId[workerNum]);
            if (rc) {
                logging::error("Error cleaning up Results Message Queue for worker %d: %m", workerNum);
                rcSave = rc;
            }
            resMqId[workerNum] = 0;
        }
    }

    return rcSave;
}

static int startDrpPython(pid_t& pyPid, unsigned workerNum, long shmemSize, const Parameters& para, DrpBase& drp)
{
    // Fork
    pyPid = vfork();

    if (pyPid == pid_t(0))
    {
        //Executing external code
        execlp("python",
               "python",
               "-u",
               "-m",
               "psdaq.drp.drp_python",
               std::to_string(para.partition).c_str(),
               std::to_string(drp.pool.pebble.bufferSize()).c_str(),
               std::to_string(para.maxTrSize).c_str(),
               std::to_string(shmemSize).c_str(),
               para.detName.c_str(),
               para.detType.c_str(),
               para.serNo.c_str(),
               std::to_string(para.detSegment).c_str(),
               std::to_string(workerNum).c_str(),
               std::to_string(para.verbose).c_str(),
               para.instrument.c_str(),
               para.prometheusDir.c_str(),
               nullptr);

        // Execlp returns only on error
        logging::critical("Error on 'execlp python' for worker %d ': %m", workerNum);
        abort();
        return 1;
    } else {
        return 0;
    }
}

int PGPDetectorApp::setupDrpPython() {

    m_shmemSize = m_drp.pool.pebble.bufferSize();
    if (m_para.maxTrSize > m_shmemSize) m_shmemSize=m_para.maxTrSize;

    // Round up to an integral number of pages
    long pageSize = sysconf(_SC_PAGESIZE);
    m_shmemSize = (m_shmemSize + pageSize - 1) & ~(pageSize - 1);

    keyBase = "p" + std::to_string(m_para.partition) + "_" + m_para.detName + "_" + std::to_string(m_para.detSegment);
    std::vector<std::thread> drpPythonThreads;

    for (unsigned workerNum=0; workerNum<m_para.nworkers; workerNum++) {

        unsigned mqSize = 512;

        // Temporary solution to start from clean msg queues and shared memory
        std::remove(("/dev/mqueue/mqinp_" + keyBase + "_" + std::to_string(workerNum)).c_str());
        std::remove(("/dev/mqueue/mqres_" + keyBase + "_" + std::to_string(workerNum)).c_str());
        std::remove(("/dev/shm/shminp_" + keyBase + "_" + std::to_string(workerNum)).c_str());
        std::remove(("/dev/shm/shmres_" + keyBase + "_" + std::to_string(workerNum)).c_str());

        // Creating message queues
        std::string key = "/mqinp_" + keyBase + "_" + std::to_string(workerNum);
        int rc = setupDrpMsgQueue(key, mqSize, m_inpMqId[workerNum], true);
        if (rc) {
            logging::error("[Thread %u] Error in creating Drp %s message queue with key %s: %m", workerNum, "Inputs", key.c_str());
            cleanupDrpPython(keyBase, m_inpMqId, m_resMqId, m_inpShmId, m_resShmId, m_para.nworkers);
            return rc;
        }
        logging::debug("[Thread %u] Created Drp msg queue %s for key %s", workerNum, "Inputs", key.c_str());
        key = "/mqres_" + keyBase + "_" + std::to_string(workerNum);
        rc = setupDrpMsgQueue(key, mqSize, m_resMqId[workerNum], false);
        if (rc) {
            logging::error("[Thread %u] Error in creating Drp %s message queue with key %s: %m", workerNum, "Inputs", key.c_str());
            cleanupDrpPython(keyBase, m_inpMqId, m_resMqId, m_inpShmId, m_resShmId, m_para.nworkers);
            return rc;
        }
        logging::debug("[Thread %u] Created Drp msg queue %s for key %s", workerNum, "Results", key.c_str());

        // Creating shared memory
        size_t shmemSize = m_drp.pool.pebble.bufferSize();
        if (m_para.maxTrSize > shmemSize) shmemSize=m_para.maxTrSize;

        // Round up to an integral number of pages
        long pageSize = sysconf(_SC_PAGESIZE);
        shmemSize = (shmemSize + pageSize - 1) & ~(pageSize - 1);

        key = "/shminp_" + keyBase + "_" + std::to_string(workerNum);
        rc = setupDrpShMem(key, shmemSize, m_inpShmId[workerNum]);
        if (rc) {
            logging::error("[Thread %u] Error in creating Drp %s shared memory for key %s: %m (open step)",
                              workerNum, "Inputs", key.c_str());
            cleanupDrpPython(keyBase, m_inpMqId, m_resMqId, m_inpShmId, m_resShmId, m_para.nworkers);
            return rc;
        }
        logging::debug("[Thread %u] Created Drp shared memory %s for key %s", workerNum, "Inputs", key.c_str());

        key = "/shmres_" + keyBase  + "_" + std::to_string(workerNum);
        rc = setupDrpShMem(key, shmemSize, m_resShmId[workerNum]);
        if (rc) {
            logging::error("[Thread %u] Error in creating Drp %s shared memory for key %s: %m (open step)",
                              workerNum, "Results", key.c_str());
            cleanupDrpPython(keyBase, m_inpMqId, m_resMqId, m_inpShmId, m_resShmId, m_para.nworkers);
            return rc;
        }
        logging::debug("[Thread %u] Created Drp shared memory %s for key %s", workerNum, "Results", key.c_str());

        logging::debug("IPC set up for worker %d", workerNum);

        startDrpPython(m_drpPids[workerNum], workerNum, shmemSize, m_para, m_drp);
    }

    logging::info("Drp python processes started");
    return 0;
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
    m_para(para),
    m_pool(para),
    m_drp(para, m_pool, context()),
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
    f.register_type<AreaDetector>    ("fakecam");
    f.register_type<AreaDetector>    ("cspad");
    f.register_type<Digitizer>       ("hsd");
    f.register_type<EpixQuad>        ("epixquad");
    f.register_type<EpixHR2x2>       ("epixhr2x2");
    f.register_type<EpixHRemu>       ("epixhremu");
    f.register_type<EpixM320>        ("epixm320");
    f.register_type<EpixUHR>         ("epixUHR");
    f.register_type<Epix100>         ("epix100");
    f.register_type<JungfrauEmulator>("jungfrauemu");
    f.register_type<Opal>            ("opal");
    f.register_type<TimeTool>        ("tt");
    f.register_type<TimingBEB>       ("tb");
    f.register_type<TimingSystem>    ("ts");
    f.register_type<Wave8>           ("wave8");
    f.register_type<HREncoder>       ("hrencoder");
    f.register_type<Piranha4>        ("piranha4");

    m_det = f.create(&m_para, &m_drp.pool);
    if (m_det == nullptr) {
        logging::critical("Error !! Could not create Detector object for %s", m_para.detType.c_str());
        throw "Could not create Detector object for " + m_para.detType;
    }

    // Initialize these to zeros. They will store the file descriptors and
    // process numbers if Drp Python is used or be just zeros if it is not.
    m_inpMqId = new int[m_para.nworkers]();
    m_resMqId = new int[m_para.nworkers]();
    m_inpShmId = new int[m_para.nworkers]();
    m_resShmId = new int[m_para.nworkers]();
    m_drpPids = new pid_t[m_para.nworkers]();

    keyBase = "";
    m_shmemSize = 0;

    auto kwargs_it = m_para.kwargs.find("drp");
    m_pythonDrp = kwargs_it != m_para.kwargs.end() && kwargs_it->second == "python";

    if (m_pythonDrp) {
        logging::info("Starting DrpPython");
        if (setupDrpPython()) {
            logging::critical("Failed to set up DrpPython");
            throw "Failed to set up DrpPython";
        }
    }

    logging::info("Ready for transitions");

    PY_RELEASE_GIL_GUARD; // Py_BEGIN_ALLOW_THREADS
}

PGPDetectorApp::~PGPDetectorApp()
{
    // Try to take things down gracefully when an exception takes us off the
    // normal path so that the most chance is given for prints to show up
    handleReset(json({}));

    if (m_pythonDrp) {
        logging::info("Cleaning up DrpPython");
        cleanupDrpPython(keyBase, m_inpMqId, m_resMqId, m_inpShmId, m_resShmId, m_para.nworkers);
    }

    if (m_drpPids)   delete [] m_drpPids;
    if (m_resShmId)  delete [] m_resShmId;
    if (m_inpShmId)  delete [] m_inpShmId;
    if (m_resMqId)   delete [] m_resMqId;
    if (m_inpMqId)   delete [] m_inpMqId;

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
        if (m_exporter)  m_exporter.reset();
        if (m_pgpThread.joinable()) {
            m_pgpThread.join();
            logging::info("PGPReader thread finished");
        }
        if (m_collectorThread.joinable()) {
            m_collectorThread.join();
            logging::info("Collector thread finished");
        }
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
            // Python-DRP is disabled during calibrations
            const std::string& config_alias = msg["body"]["config_alias"];
            bool pythonDrp = config_alias != "CALIB" ? m_pythonDrp : false;

            m_pgpDetector = std::make_unique<PGPDetector>(m_para, m_drp, m_det, pythonDrp, m_inpMqId,
                                                          m_resMqId, m_inpShmId, m_resShmId, m_shmemSize);
            m_exporter = std::make_shared<Pds::MetricExporter>();
            if (m_drp.exposer()) {
                m_drp.exposer()->RegisterCollectable(m_exporter);
            }

            m_pgpThread = std::thread{&PGPDetector::reader, std::ref(*m_pgpDetector), m_exporter,
                                      std::ref(m_det), std::ref(m_drp.tebContributor())};
            m_collectorThread = std::thread(&PGPDetector::collector, std::ref(*m_pgpDetector),
                                            std::ref(m_drp.tebContributor()));

            // Provide EbReceiver with the Detector interface so that additional
            // data blocks can be formatted into the XTC, e.g. trigger information
            m_drp.ebReceiver().configure(m_det, m_pgpDetector.get());

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

    if (m_pythonDrp) {
        drainDrpMessageQueues();
        if (msg != json({})) // Skip this when exiting since python is already gone
            resetDrpPython();
    }

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

void PGPDetectorApp::connectionShutdown()
{
    if (m_det) {
        m_det->connectionShutdown();
    }

    m_drp.shutdown();
    if (m_exporter) {
        m_exporter.reset();
    }
}

void PGPDetectorApp::drainDrpMessageQueues()
{
    // Drains the message queues to make sure that no
    // undelivered message is in them.
    char recvmsg[520];

    for (unsigned workerNum=0; workerNum<m_para.nworkers; workerNum++) {
        if (m_inpMqId[workerNum]) {
            [[maybe_unused]] int rc = drpRecv(m_inpMqId[workerNum], recvmsg, sizeof(recvmsg), 0);
        }
    }

    for (unsigned workerNum=0; workerNum<m_para.nworkers; workerNum++) {
        if (m_resMqId[workerNum]) {
            [[maybe_unused]] int rc = drpRecv(m_resMqId[workerNum], recvmsg, sizeof(recvmsg), 0);
        }
    }
}

int PGPDetectorApp::resetDrpPython()
{
    int rcSave = 0;
    char recvmsg[520];

    for (unsigned workerNum=0; workerNum<m_para.nworkers; workerNum++) {
        int rc = drpSend(m_inpMqId[workerNum], "s", 1);
        if (rc) {
            logging::error("Error sending reset message to Drp python worker %u: %m",
                           workerNum);
            rcSave = rc;
            continue;
        }
        rc = drpRecv(m_resMqId[workerNum], recvmsg, sizeof(recvmsg), 10000);
        if (rc) {
            logging::error("Error receiving reset message from Drp python worker %u: %m",
                           workerNum);
            rcSave = rc;
        }
    }

    return rcSave;
}

}
