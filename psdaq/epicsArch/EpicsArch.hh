#pragma once

#include <thread>
#include <atomic>
#include <string>
#include "drp/DrpBase.hh"
#include "drp/XpmDetector.hh"
#include "psdaq/service/Collection.hh"
#include "EpicsArchMonitor.hh"


namespace Drp {

class Pgp : public PgpReader
{
public:
    Pgp(const Parameters& para, DrpBase& drp, Detector* det, const bool& running);
    Pds::EbDgram* next(uint32_t& evtIndex);
    const uint64_t nDmaRet() { return m_nDmaRet; }
private:
    Pds::EbDgram* _handle(uint32_t& evtIndex);
    Detector* m_det;
    Pds::Eb::TebContributor& m_tebContributor;
    static const int MAX_RET_CNT_C = 100;
    const bool& m_running;
    int32_t m_available;
    int32_t m_current;
    unsigned m_nodeId;
    uint64_t m_nDmaRet;
};


class EaDetector : public XpmDetector
{
public:
    EaDetector(Parameters& para, const std::string& pvCfgFile, DrpBase& drp);
    ~EaDetector();
    unsigned connect(std::string& msg);
    unsigned disconnect();
    unsigned unconfigure();
    const PgpReader* pgp() { return &m_pgp; }
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
    Pgp m_pgp;
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
    nlohmann::json connectionInfo(const nlohmann::json& msg) override;
    void connectionShutdown() override;
    void handleConnect(const nlohmann::json& msg) override;
    void handleDisconnect(const nlohmann::json& msg) override;
    void handlePhase1(const nlohmann::json& msg) override;
    void _unconfigure();
    void _disconnect();
    void _error(const std::string& which, const nlohmann::json& msg, const std::string& errorMsg);
private:
    Drp::Parameters& m_para;
    MemPoolCpu m_pool;
    DrpBase m_drp;
    std::unique_ptr<EaDetector> m_eaDetector;
    Detector* m_det;
    bool m_unconfigure;
};

} // EpicsArch
