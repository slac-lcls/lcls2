#pragma once

#include <utility>
#include <memory>
#include <string>
#include <thread>

#include "drp.hh"
#include <Python.h>
#include "DrpBase.hh"
#include "GpuDetector.hh"
#include "GpuWorker.hh"
#include "psdaq/trigger/TriggerPrimitive.hh"
#include "psdaq/service/Collection.hh"
#include <nlohmann/json.hpp>
#include "psdaq/service/Dl.hh"
#include "psdaq/service/MetricExporter.hh"

namespace Drp {

class GpuDetector;


class GpuWorkerFactory : public Pds::Dl
{
public:
    //template <typename T>  // @todo: Always GpuWorker, I think
    void register_type(const std::string& name, const std::string& solib)
    {
        // @todo: Not needed anymore, I think
        //static_assert(std::is_base_of<GpuWorker, T>::value,
        //              //"Dl::instantiate: T (" T ") must be derived from GpuWorker");
        //              "Dl::instantiate: T must be derived from GpuWorker");

        // This makes a copy of the solib string in the map so it's available for create()
        m_create_funcs.emplace(name, std::make_pair(&_instantiate, solib));
    }

    Detector* create(const std::string& name, Parameters& para, MemPool& pool)
    {
        auto it = m_create_funcs.find(name);
        if (it != m_create_funcs.end()) {
            auto pair = it->second;
            return pair.first(this, pair.second, "createDetectorGpu", para, pool);
        }
        return nullptr;
    }

private:
    static
    Detector* _instantiate(Pds::Dl* dl,
                            const std::string& library, const std::string& symbol,
                            Parameters& para, MemPool& pool)
    {
        printf("Loading object symbols from library '%s'\n", library.c_str());

        if (dl->open(library, RTLD_LAZY))
        {
            fprintf(stderr, "%s:\n  Error opening library '%s'\n",
                    __PRETTY_FUNCTION__, library.c_str());
            return nullptr;
        }

        typedef Detector* fn_t(Parameters& para, MemPool& pool);
        fn_t* instantiateFn = reinterpret_cast<fn_t*>(dl->loadSymbol(symbol.c_str()));
        if (!instantiateFn)
        {
            fprintf(stderr, "%s:\n  Symbol '%s' not found in %s\n",
                    __PRETTY_FUNCTION__, symbol.c_str(), library.c_str());
            return nullptr;
        }
        Detector* instance = instantiateFn(para, pool);
        if (!instance)
        {
            fprintf(stderr, "%s:\n  Error calling %s\n",
                    __PRETTY_FUNCTION__, symbol.c_str());
            return nullptr;
        }
        return instance;
    }

private:
    typedef Detector* (*PCreateFunc)(Pds::Dl*,
                                     const std::string& so, const std::string& sym,
                                     Parameters& para, MemPool& pool);
    std::unordered_map<std::string, std::pair<PCreateFunc, const std::string> > m_create_funcs;
};


class GpuDetectorApp : public CollectionApp
{
public:
    GpuDetectorApp(Parameters& para);
    virtual ~GpuDetectorApp();
    void initialize();
private:
    nlohmann::json connectionInfo(const nlohmann::json& msg) override;
    void connectionShutdown() override;
    void handleReset(const nlohmann::json& msg) override;
    void handleConnect(const nlohmann::json& msg) override;
    void handleDisconnect(const nlohmann::json& msg) override;
    void handlePhase1(const nlohmann::json& msg) override;
    void _unconfigure();
    void _disconnect();
private:
    Parameters&                          m_para;
    MemPoolGpu                           m_pool;
    DrpBase                              m_drp;
    GpuWorkerFactory                     m_factory;
    Detector*                            m_det;
    std::thread                          m_collectorThread;
    std::unique_ptr<GpuDetector>         m_gpuDetector;
    std::shared_ptr<Pds::MetricExporter> m_exporter;
    bool                                 m_unconfigure;
    PyThreadState*                       m_pysave;
};

}
