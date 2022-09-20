#pragma once

#include <thread>
#include <Python.h>
#include "DrpBase.hh"
#include "PGPDetector.hh"
#include "psdaq/trigger/TriggerPrimitive.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/service/MetricExporter.hh"

namespace Drp {

class PGPDetectorApp : public CollectionApp
{
public:
    PGPDetectorApp(Parameters& para);
    virtual ~PGPDetectorApp();
    void initialize();
    nlohmann::json connectionInfo() override;
    void connectionShutdown() override;
    void handleReset(const nlohmann::json& msg) override;
private:
    void handleConnect(const nlohmann::json& msg) override;
    void handleDisconnect(const nlohmann::json& msg) override;
    void handlePhase1(const nlohmann::json& msg) override;
    void unconfigure();
    void disconnect();
    DrpBase m_drp;
    Parameters& m_para;
    std::thread m_pgpThread;
    std::thread m_collectorThread;
    std::unique_ptr<PGPDetector> m_pgpDetector;
    Detector* m_det;
    std::shared_ptr<Pds::MetricExporter> m_exporter;
    bool m_unconfigure;
    PyThreadState*    m_pysave;
};

}
