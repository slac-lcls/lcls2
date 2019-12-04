#include <iostream>
#include <iomanip>
#include "drp.hh"
#include "Detector.hh"
#include "TimingSystem.hh"
#include "TimeTool.hh"
#include "AreaDetector.hh"
#include "Digitizer.hh"
#include "psdaq/service/MetricExporter.hh"
#include "PGPDetectorApp.hh"
#include "psalg/utils/SysLog.hh"
#include "RunInfoDef.hh"


using json = nlohmann::json;
using logging = psalg::SysLog;

namespace Drp {

PGPDetectorApp::PGPDetectorApp(Parameters& para) :
    CollectionApp(para.collectionHost, para.partition, "drp", para.alias),
    m_drp(para, context()),
    m_para(para)
{
    Factory<Detector> f;
    f.register_type<TimingSystem>("TimingSystem");
    f.register_type<Digitizer>("Digitizer");
    f.register_type<TimeTool>("TimeTool");
    f.register_type<AreaDetector>("AreaDetector");
    m_det = f.create(&m_para, &m_drp.pool);
    if (m_det == nullptr) {
        logging::error("Error !! Could not create Detector object");
    }
    if (m_para.outputDir.empty()) {
        logging::info("output dir: n/a");
    } else {
        logging::info("output dir: %s", m_para.outputDir.c_str());
    }
    logging::info("Ready for transitions");
}

void PGPDetectorApp::shutdown()
{
    m_exporter.reset();

    if (m_pgpDetector) {
        m_pgpDetector->shutdown();
        if (m_pgpThread.joinable()) {
            m_pgpThread.join();
        }
        if (m_collectorThread.joinable()) {
            m_collectorThread.join();
        }
        m_pgpDetector.reset();
    }
    m_drp.shutdown();

    m_det->namesLookup().clear();   // erase all elements
}

void PGPDetectorApp::handleConnect(const json& msg)
{
    json body = json({});
    std::string errorMsg = m_drp.connect(msg, getId());
    if (!errorMsg.empty()) {
        logging::error("Error in DrpBase::connect");
        logging::error("%s", errorMsg.c_str());
        body["err_info"] = errorMsg;
    }

    m_det->nodeId = m_drp.nodeId();
    m_det->connect(msg, std::to_string(getId()));

    m_unconfigure = false;
    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void PGPDetectorApp::handleDisconnect(const json& msg)
{
    shutdown();

    json body = json({});
    reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void PGPDetectorApp::handlePhase1(const json& msg)
{
    logging::debug("handlePhase1 in PGPDetectorApp");

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

        m_exporter = std::make_shared<MetricExporter>();
        if (m_drp.exposer()) {
            m_drp.exposer()->RegisterCollectable(m_exporter);
        }

        m_pgpThread = std::thread{&PGPDetector::reader, std::ref(*m_pgpDetector), m_exporter,
                                  std::ref(m_drp.tebContributor())};
        m_collectorThread = std::thread(&PGPDetector::collector, std::ref(*m_pgpDetector),
                                        std::ref(m_drp.tebContributor()));

        std::string config_alias = msg["body"]["config_alias"];
        unsigned error = m_det->configure(config_alias, xtc);

        m_drp.runInfoSupport(xtc, m_det->namesLookup());

        if (error) {
            std::string errorMsg = "Phase 1 error in Detector::configure";
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
        m_pgpDetector->resetEventCounter();
    }
    else if (key == "unconfigure") {
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
    }

    json answer = createMsg(key, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void PGPDetectorApp::handleReset(const json& msg)
{
    shutdown();
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
