#pragma once

#include <utility>
#include <memory>
#include <string>
#include <unordered_map>
#include <thread>

#include "drp/drp.hh"
#include <Python.h>
#include "drp/DrpBase.hh"
#include "psdaq/service/Collection.hh"
#include <nlohmann/json.hpp>
#include "psdaq/service/Dl.hh"
#include "psdaq/service/MetricExporter.hh"
#include "MemPool.hh"
#include "Detector.hh"

namespace Drp {
  namespace Gpu {

class PGPDetector;

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
    Parameters&                          m_para;
    MemPoolGpu                           m_pool;
    DrpBase                              m_drp;
    DetectorFactory                      m_factory;
    Detector*                            m_det;
    std::thread                          m_collectorThread;
    std::unique_ptr<PGPDetector>         m_pgpDetector;
    std::shared_ptr<Pds::MetricExporter> m_exporter;
    bool                                 m_unconfigure;
    PyThreadState*                       m_pysave;
};

  } // Gpu
} // Drp
