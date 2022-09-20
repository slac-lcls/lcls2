#pragma once

#include <thread>
#include <atomic>
#include <string>
#include "drp/DrpBase.hh"
#include "drp/XpmDetector.hh"
#include "psdaq/service/Collection.hh"
#include "EpicsArchMonitor.hh"


namespace Drp {

class EaDetector : public XpmDetector
{
public:
    EaDetector(Parameters& para, const std::string& pvCfgFile, DrpBase& drp);
    ~EaDetector();
    unsigned connect(std::string& msg);
    unsigned disconnect();
    unsigned unconfigure();
public:                                 // Detector virtuals
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) override;
    void slowupdate(XtcData::Xtc& xtc, const void* bufEnd) override;
    void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event) override;
private:
    void _worker();
    void _sendToTeb(Pds::EbDgram& dgram, uint32_t index);
private:
    const std::string& m_pvCfgFile;
    DrpBase& m_drp;
    std::unique_ptr<EpicsArchMonitor> m_monitor;
    std::thread m_workerThread;
    std::atomic<bool> m_terminate;
    bool m_running;
    std::shared_ptr<Pds::MetricExporter> m_exporter;
    uint64_t m_nEvents;
    uint64_t m_nUpdates;
    uint64_t m_nStales;
};


class EpicsArchApp : public CollectionApp
{
public:
    EpicsArchApp(Drp::Parameters& para, const std::string& pvCfgFile);
    ~EpicsArchApp();
    void handleReset(const nlohmann::json& msg) override;
private:
    nlohmann::json connectionInfo() override;
    void connectionShutdown() override;
    void handleConnect(const nlohmann::json& msg) override;
    void handleDisconnect(const nlohmann::json& msg) override;
    void handlePhase1(const nlohmann::json& msg) override;
    void _unconfigure();
    void _disconnect();
    void _error(const std::string& which, const nlohmann::json& msg, const std::string& errorMsg);
private:
    DrpBase m_drp;
    Drp::Parameters& m_para;
    std::unique_ptr<EaDetector> m_eaDetector;
    Detector* m_det;
    bool m_unconfigure;
};

} // EpicsArch
