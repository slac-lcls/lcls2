#include <iostream>
#include "drp.hh"
#include "Detector.hh"
#include "TimingSystem.hh"
#include "TimeTool.hh"
#include "AreaDetector.hh"
#include "Digitizer.hh"
#include "psdaq/service/MetricExporter.hh"
#include "PGPDetectorApp.hh"

using json = nlohmann::json;

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
        std::cout<< "Error !! Could not create Detector object\n";
    }
    std::cout << "output dir: " << m_para.outputDir << std::endl;
}

void PGPDetectorApp::shutdown()
{
    if (m_pgpDetector) {
        m_pgpDetector->shutdown();
        if (m_pgpThread.joinable()) {
            m_pgpThread.join();
        }
        if (m_collectorThread.joinable()) {
            m_collectorThread.join();
        }
    }
    m_drp.shutdown();
}

void PGPDetectorApp::handleConnect(const json& msg)
{
    json body = json({});
    std::string errorMsg = m_drp.connect(msg, getId());
    if (!errorMsg.empty()) {
        std::cout<<"Error in DrpBase::connect\n";
        std::cout<<errorMsg<<'\n';
        body["err_info"] = errorMsg;
    }

    m_det->nodeId = m_drp.nodeId();
    m_det->connect(msg, std::to_string(getId()));

    m_pgpDetector = std::make_unique<PGPDetector>(m_para, m_drp.pool, m_det);

    auto exporter = std::make_shared<MetricExporter>();
    if (m_drp.exposer()) {
        m_drp.exposer()->RegisterCollectable(exporter);
    }
    m_pgpThread = std::thread{&PGPDetector::reader, std::ref(*m_pgpDetector), exporter};
    m_collectorThread = std::thread(&PGPDetector::collector, std::ref(*m_pgpDetector),
                                    std::ref(m_drp.tebContributor()));

    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void PGPDetectorApp::handleDisconnect(const json& msg)
{
    m_drp.disconnect(msg);
    shutdown();
    json body = json({});
    reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void PGPDetectorApp::handlePhase1(const json& msg)
{
    std::cout<<"handlePhase1 in DrpApp\n";

    std::string key = msg["header"]["key"];
    unsigned error = 0;
    if (key == "configure") {
        std::string config_alias = msg["body"]["config_alias"];
        XtcData::Xtc& xtc = m_det->transitionXtc();
        XtcData::TypeId tid(XtcData::TypeId::Parent, 0);
        xtc.contains = tid;
        xtc.damage = 0;
        xtc.extent = sizeof(XtcData::Xtc);
        error = m_det->configure(config_alias, xtc);
        m_pgpDetector->resetEventCounter();
    }

    json answer;
    json body = json({});
    if (error) {
        body["err_info"] = "phase 1 error";
        std::cout<<"transition phase1 error\n";
    }
    else {
        std::cout<<"transition phase1 complete\n";
    }
    answer = createMsg(msg["header"]["key"], msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void PGPDetectorApp::handleReset(const json& msg)
{
}

json PGPDetectorApp::connectionInfo()
{
    std::string ip = getNicIp();
    std::cout<<"nic ip  "<<ip<<'\n';
    json body = {{"connect_info", {{"nic_ip", ip}}}};
    json info = m_det->connectionInfo();
    body["connect_info"].update(info);
    json bufInfo = m_drp.connectionInfo();
    body["connect_info"].update(bufInfo);
    return body;
}

}
