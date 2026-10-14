#pragma once

#include <thread>
#include <Python.h>
#include "DrpBase.hh"
#include "PGPDetector.hh"
#include "psdaq/service/Collection.hh"

namespace Drp {

class PGPDetectorApp : public CollectionApp
{
public:
    PGPDetectorApp(Parameters& para);
    virtual ~PGPDetectorApp();
    void initialize();
    nlohmann::json connectionInfo(const nlohmann::json& msg) override;
    void connectionShutdown() override;
    void handleReset(const nlohmann::json& msg) override;
private:
    void handleDealloc(const nlohmann::json& msg) override;
    void handleConnect(const nlohmann::json& msg) override;
    void handleDisconnect(const nlohmann::json& msg) override;
    void handlePhase1(const nlohmann::json& msg) override;
    int setupDrpPython();
    void unconfigure();
    void disconnect();
    void drainDrpMessageQueues();
    int resetDrpPython();
    Parameters& m_para;
    MemPoolCpu m_pool;
    Detector* m_det;
    std::unique_ptr<PGPDrp> m_drp;
    bool m_unconfigure;
    PyThreadState* m_pysave;
    int* m_inpMqId;
    int* m_resMqId;
    int* m_inpShmId;
    int* m_resShmId;
    std::string keyBase;
    pid_t* m_drpPids;
    size_t m_shmemSize;
    bool m_pythonDrp;
};

}
