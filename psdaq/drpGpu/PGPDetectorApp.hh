#pragma once

#include <utility>
#include <memory>
#include <string>
#include <unordered_map>
#include <thread>
#include <Python.h>

#include "drp/drp.hh"
#include "drp/DrpBase.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/service/Dl.hh"
#include "MemPool.hh"
#include "Detector.hh"

namespace Drp {
  namespace Gpu {

class PGPDrp;

class DetectorFactory
{
public:
    void register_type(const std::string& name, const std::string& solib);
    Drp::Gpu::Detector* create(const std::string& name, Parameters& para, MemPoolGpu& pool);
private:
    static
    Drp::Gpu::Detector* _instantiate(Pds::Dl& dl, const std::string& soName,
                                     Parameters& para, MemPoolGpu& pool);
private:
    Pds::Dl m_dl;
    std::unordered_map<std::string, std::string> m_create_funcs;
};

class PGPDetectorApp : public CollectionApp
{
public:
    PGPDetectorApp(Parameters& para);
    virtual ~PGPDetectorApp();
    void initialize();
private:
    nlohmann::json connectionInfo(const nlohmann::json& msg) override;
    void connectionShutdown() override;
    void handleReset(const nlohmann::json& msg) override;
private:
    void handleDealloc(const nlohmann::json& msg) override;
    void handleConnect(const nlohmann::json& msg) override;
    void handleDisconnect(const nlohmann::json& msg) override;
    void handlePhase1(const nlohmann::json& msg) override;
    void _unconfigure();
    void _disconnect();
private:
    Parameters&             m_para;
    MemPoolGpu              m_pool;
    Detector*               m_det;
    std::unique_ptr<PGPDrp> m_drp;
    bool                    m_unconfigure;
    PyThreadState*          m_pysave;
    DetectorFactory         m_factory;
};

  } // Gpu
} // Drp
