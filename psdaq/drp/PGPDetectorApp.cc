#include <iostream>
#include <iomanip>
#include "drp.hh"
#include "Detector.hh"
#include "TimingSystem.hh"
#include "TimeTool.hh"
#include "AreaDetector.hh"
#include "Digitizer.hh"
#include "Opal.hh"
#include "Wave8.hh"
#include "psdaq/service/MetricExporter.hh"
#include "PGPDetectorApp.hh"
#include "psalg/utils/SysLog.hh"
#include "RunInfoDef.hh"

#define PY_RELEASE_GIL    PyEval_SaveThread()
#define PY_ACQUIRE_GIL(x) PyEval_RestoreThread(x)
//#define PY_RELEASE_GIL    0
//#define PY_ACQUIRE_GIL(x) {}

using json = nlohmann::json;
using logging = psalg::SysLog;

namespace Drp {

PGPDetectorApp::PGPDetectorApp(Parameters& para) :
    CollectionApp(para.collectionHost, para.partition, "drp", para.alias),
    m_drp(para, context()),
    m_para(para)
{
    Py_Initialize(); // for use by configuration

    Factory<Detector> f;
    f.register_type<TimingSystem>("ts");
    f.register_type<Digitizer>("hsd");
    f.register_type<AreaDetector>("fakecam");
    f.register_type<AreaDetector>("cspad");
    f.register_type<TimeTool>("tt");
    f.register_type<Wave8>("wave8");
    f.register_type<Opal>("opal");

    m_det = f.create(&m_para, &m_drp.pool);
    if (m_det == nullptr) {
        logging::error("Error !! Could not create Detector object for %s", m_para.detType);
        throw "Could not create Detector object for " + m_para.detType;
    }
    if (m_para.outputDir.empty()) {
        logging::info("output dir: n/a");
    } else {
        logging::info("output dir: %s", m_para.outputDir.c_str());
    }
    logging::info("Ready for transitions");

    m_pysave = PY_RELEASE_GIL; // Py_BEGIN_ALLOW_THREADS
}

PGPDetectorApp::~PGPDetectorApp()
{
    try {
        PyGILState_Ensure();
        Py_Finalize(); // for use by configuration
    } catch(const std::exception &e) {
        std::cout << "Exception in ~PGPDetectorApp(): " << e.what() << "\n";
    } catch(...) {
        std::cout << "Exception in Python code: UNKNOWN\n";
    }
}

void PGPDetectorApp::shutdown()
{
    if (m_det) {
        m_det->shutdown();
    }

    if (m_pgpDetector) {
        m_drp.stop();                   // Release allocate()
        m_pgpDetector->shutdown();
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
    m_drp.shutdown();

    m_det->namesLookup().clear();   // erase all elements
}

void PGPDetectorApp::handleConnect(const json& msg)
{
    PY_ACQUIRE_GIL(m_pysave);  // Py_END_ALLOW_THREADS

    json body = json({});
    std::string errorMsg = m_drp.connect(msg, getId());
    if (!errorMsg.empty()) {
        logging::error("Error in DrpBase::connect");
        logging::error("%s", errorMsg.c_str());
        body["err_info"] = errorMsg;
    }

    m_det->nodeId = m_drp.nodeId();
    m_det->connect(msg, std::to_string(getId()));

    m_pysave = PY_RELEASE_GIL; // Py_BEGIN_ALLOW_THREADS

    m_unconfigure = false;
    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void PGPDetectorApp::handleDisconnect(const json& msg)
{
    PY_ACQUIRE_GIL(m_pysave);  // Py_END_ALLOW_THREADS

    shutdown();

    m_pysave = PY_RELEASE_GIL; // Py_BEGIN_ALLOW_THREADS

    json body = json({});
    reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void PGPDetectorApp::handlePhase1(const json& msg)
{
    logging::debug("handlePhase1 in PGPDetectorApp");

    PY_ACQUIRE_GIL(m_pysave);  // Py_END_ALLOW_THREADS

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
    std::string key = msg["header"]["key"];

    if (key == "configure") {
        if (m_unconfigure) {
            shutdown();
            m_unconfigure = false;
        }

        std::string errorMsg = m_drp.configure(msg);
        if (!errorMsg.empty()) {
            errorMsg = "Phase 1 error: " + errorMsg;
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }

        m_pgpDetector = std::make_unique<PGPDetector>(m_para, m_drp, m_det);

        if (m_exporter)  m_exporter.reset();
        m_exporter = std::make_shared<Pds::MetricExporter>();
        if (m_drp.exposer()) {
            m_drp.exposer()->RegisterCollectable(m_exporter);
        }

        m_pgpThread = std::thread{&PGPDetector::reader, std::ref(*m_pgpDetector), m_exporter,
                                  std::ref(m_det), std::ref(m_drp.tebContributor())};
        m_collectorThread = std::thread(&PGPDetector::collector, std::ref(*m_pgpDetector),
                                        std::ref(m_drp.tebContributor()));
        std::string config_alias = msg["body"]["config_alias"];
        unsigned error = m_det->configure(config_alias, xtc);
        if (error) {
            std::string errorMsg = "Phase 1 error in Detector::configure";
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
        else {
            m_drp.runInfoSupport(xtc, m_det->namesLookup());
        }
        m_pgpDetector->resetEventCounter();
    }
    else if (key == "unconfigure") {
        // Delay unconfiguration until after phase 2 of unconfigure has completed
        m_unconfigure = true;
    }
    else if (key == "beginstep") {
        // see if we find some step information in phase 1 that needs to be
        // to be attached to the xtc
        m_det->beginstep(xtc, phase1Info);
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
        m_pgpDetector->resetEventCounter();
    }
    else if (key == "endrun") {
        std::string errorMsg = m_drp.endrun(phase1Info);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
    }

    m_pysave = PY_RELEASE_GIL; // Py_BEGIN_ALLOW_THREADS

    json answer = createMsg(key, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void PGPDetectorApp::handleReset(const json& msg)
{
    PY_ACQUIRE_GIL(m_pysave);  // Py_END_ALLOW_THREADS

    shutdown();
    m_drp.reset();
    if (m_exporter)  m_exporter.reset();

    m_pysave = PY_RELEASE_GIL; // Py_BEGIN_ALLOW_THREADS
}

json PGPDetectorApp::connectionInfo()
{
    std::string ip = getNicIp();
    logging::debug("nic ip  %s", ip.c_str());
    json body = {{"connect_info", {{"nic_ip", ip}}}};
    json info = m_det->connectionInfo();
    body["connect_info"].update(info);
    json bufInfo = m_drp.connectionInfo();
    body["connect_info"].update(bufInfo); // Revisit: Should be in det_info
    return body;
}

}
