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
    Pgp(const Parameters& para, MemPool& pool, Detector* det);
    Pds::EbDgram* next(uint32_t& evtIndex);
    const uint64_t nDmaRet() const { return m_nDmaRet; }
private:
    Pds::EbDgram* _handle(uint32_t& evtIndex);
    Detector*        m_det;
    static const int MAX_RET_CNT_C = 100;
    int32_t          m_available;
    int32_t          m_current;
    uint64_t         m_nDmaRet;
};


class EaDetector : public XpmDetector
{
public:
    EaDetector(Parameters&, const std::string& pvCfgFile, MemPoolCpu&);
    virtual ~EaDetector();
    unsigned connect(const nlohmann::json&, const std::string& collectionId, std::string& msg);
    unsigned disconnect();
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) override;
    unsigned unconfigure();
    void slowupdate(XtcData::Xtc& xtc, const void* bufEnd) override;
    void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t l1count) override;
public:
    const std::unique_ptr<EpicsArchMonitor>& monitor() const { return m_monitor; }
    const uint64_t nStales() const { return m_nStales; }
private:
    enum {RawNamesIndex = NamesIndex::BASE, InfoNamesIndex};
    const std::string&                m_pvCfgFile;
    std::unique_ptr<EpicsArchMonitor> m_monitor;
    uint64_t                          m_nStales;
};



class EaDrp : public DrpBase
{
public:
    EaDrp(Parameters&, MemPoolCpu&, Detector&, ZmqContext&);
    virtual ~EaDrp() {}
    std::string configure(const nlohmann::json& msg);
    unsigned unconfigure();
private:
    int  _setupMetrics(const std::shared_ptr<Pds::MetricExporter>);
    void _worker();
    void _sendToTeb(const Pds::EbDgram& dgram, uint32_t index);
private:
    const Parameters& m_para;
    Detector&         m_det;
    Pgp               m_pgp;
    std::thread       m_workerThread;
    std::atomic<bool> m_terminate;
    uint64_t          m_nEvents;
    uint64_t          m_nUpdates;
};


class EpicsArchApp : public CollectionApp
{
public:
    EpicsArchApp(Parameters& para, const std::string& pvCfgFile);
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
    Parameters&                 m_para;
    MemPoolCpu                  m_pool;
    std::unique_ptr<EaDetector> m_det;
    std::unique_ptr<EaDrp>      m_drp;
    bool                        m_unconfigure;
};

} // Drp
