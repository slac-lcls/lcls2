#pragma once

#include <thread>
#include <atomic>
#include <string>
#include <functional>
#include <mutex>
#include <condition_variable>
#include "DrpBase.hh"
#include "PGPDetector.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/epicstools/PVBase.hh"

namespace Drp {

class PvaApp;

class PvaMonitor : public Pds_Epics::PVBase
{
public:
    PvaMonitor(const char* channelName, PvaApp& app) : Pds_Epics::PVBase(channelName), m_app(app) {}
    void printStructure();
    XtcData::VarDef get(size_t& payloadSize);
    void updated() override;
public:
    std::function<size_t(void* data, size_t& length)> getData;
private:
    template<typename T> size_t _getDatumT(void* data, size_t& length) {
        *static_cast<T*>(data) = getScalarAs<T>();
        length = 1;
        return sizeof(T);
    }
  template<typename T> size_t _getDataT(void* data, size_t& length) {
        //pvd::shared_vector<const T> vec((T*)data, [](void*){}, 0, 128); // Doesn't work
        pvd::shared_vector<const T> vec;
        getVectorAs<T>(vec);
        length = vec.size();
        size_t size = length * sizeof(T);
        memcpy(data, vec.data(), size);
        return size;
    }
private:
    PvaApp& m_app;
};

class PvaApp : public CollectionApp
{
public:
  PvaApp(Parameters& para, const std::string& pvaName);
    void process(const PvaMonitor&);
    void handleReset(const nlohmann::json& msg) override;
private:
    nlohmann::json connectionInfo() override;
    void handleConnect(const nlohmann::json& msg) override;
    void handleDisconnect(const nlohmann::json& msg) override;
    void handlePhase1(const nlohmann::json& msg) override;
    void _connectPgp(const nlohmann::json& json, const std::string& collectionId);
    void _worker(std::shared_ptr<MetricExporter> exporter);
    void _sendToTeb(Pds::EbDgram& dgram, uint32_t index);
    void _shutdown();
    void _error(const std::string& which, const nlohmann::json& msg, const std::string& errorMsg);

    enum { PvaNamesIndex = NamesIndex::BASE};
    DrpBase m_drp;
    Parameters& m_para;
    RunInfo m_runInfo;
    const std::string& m_pvName;
    std::thread m_workerThread;
    SPSCQueue<uint32_t> m_inputQueue;
    std::unique_ptr<PvaMonitor> m_pvaMonitor;
    XtcData::NamesLookup m_namesLookup;
    mutable std::mutex _lock;
    std::condition_variable _cv;
    std::atomic<bool> m_swept;
    std::atomic<bool> m_terminate;
    std::shared_ptr<MetricExporter> m_exporter;
    bool m_unconfigure;
    uint64_t m_nEvents;
    uint64_t m_nUpdates;
    uint64_t m_nMissed;
    uint64_t m_nEmpty;
    uint64_t m_nTooOld;
};

}
