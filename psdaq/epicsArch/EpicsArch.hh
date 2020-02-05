#pragma once

#include <thread>
#include <atomic>
#include <string>
#include <functional>
#include <mutex>
#include <condition_variable>
#include "drp/DrpBase.hh"
#include "drp/drp.hh"
#include "psdaq/service/Collection.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "EpicsArchMonitor.hh"


namespace Drp {

class EpicsArchApp : public CollectionApp
{
public:
    EpicsArchApp(Drp::Parameters& para, const std::string& pvCfgFile);
    ~EpicsArchApp();
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

    enum { MyNamesIndex = NamesIndex::BASE};
    DrpBase m_drp;
    Drp::Parameters& m_para;
    RunInfo m_runInfo;
    const std::string& m_pvCfgFile;
    std::thread m_workerThread;
    std::unique_ptr<Pds::EpicsArchMonitor> m_monitor;
    XtcData::NamesLookup m_namesLookup;
    std::atomic<bool> m_terminate;
    std::shared_ptr<MetricExporter> m_exporter;
    bool m_unconfigure;
    uint64_t m_nEvents;
    uint64_t m_nUpdates;
    uint64_t m_nConnected;
};

} // EpicsArch
